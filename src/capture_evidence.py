"""Capture actual public pages for the final submission; requires Playwright locally."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/harshitrathi14/capstone-predictivemaintenance'


def capture(app_url, workflow_record):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.inference import load_bundle, predict_observation
    model, profile, _ = load_bundle()
    values = {name: round(profile['summary'][name]['median'], 2) for name in profile['features']}
    _, prediction, score, _ = predict_observation(values, model, profile)
    out = ROOT/'evidence'
    out.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width':1280,'height':1080},device_scale_factor=1)
        page.goto(REPO,wait_until='domcontentloaded',timeout=60000)
        page.locator('a[href$="/streamlit_app.py"]:visible').first.wait_for(timeout=30000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out/'github_repository.png'),full_page=False)
        page.screenshot(path=str(out/'github_repository_report.png'),
                        clip={'x':24,'y':180,'width':925,'height':520})
        page.goto(workflow_record['url'],wait_until='domcontentloaded',timeout=60000)
        page.get_by_text('persist-model',exact=True).first.wait_for(timeout=30000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out/'github_workflow.png'),full_page=False)
        page.screenshot(path=str(out/'github_workflow_report.png'),
                        clip={'x':24,'y':180,'width':1232,'height':440})
        page.set_viewport_size({'width':1440,'height':1180})
        page.goto(app_url,wait_until='domcontentloaded',timeout=60000)
        # Community Cloud may expose the app in an iframe.
        app = None
        for _ in range(60):
            for frame in page.frames:
                if frame.get_by_role('button',name='Assess condition',exact=True).count():
                    app = frame
                    break
            if app is not None:
                break
            page.wait_for_timeout(1000)
        if app is None:
            page.screenshot(path=str(out/'deployment_unavailable.png'),full_page=True)
            raise RuntimeError('The public app did not display its prediction form.')
        app.get_by_role('button',name='Assess condition',exact=True).click()
        app.get_by_text(f'Class {prediction}',exact=True).wait_for(timeout=60000)
        app.get_by_text(f'{score:.2%}',exact=True).wait_for(timeout=30000)
        app.get_by_role('button',name='Download assessment CSV').wait_for(timeout=30000)
        # Allow charts/dataframe and web-font rendering to settle after the streamed response.
        page.wait_for_timeout(2000)
        page.screenshot(path=str(out/'streamlit_cloud.png'),full_page=True)
        actual_url = page.url
        browser.close()
    record = {
        'captured_at': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
        'repository_url': REPO,
        'workflow': workflow_record,
        'deployment': {'url':app_url, 'resolved_url':actual_url, 'verified':True,
                       'inputs':values,'predicted_class':prediction,'class_1_score':score,
                       'model_sha256':profile['model_sha256'],
                       'check':'Submitted default inputs on the public app; displayed class and rounded score match committed-model inference.'}}
    (out/'final_evidence.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('app_url')
    parser.add_argument('--workflow-record',default='evidence/workflow_record.json')
    args=parser.parse_args()
    capture(args.app_url,json.loads((ROOT/args.workflow_record).read_text()))
