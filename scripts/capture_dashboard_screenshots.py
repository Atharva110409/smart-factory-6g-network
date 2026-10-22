import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def capture_screenshots():
    os.makedirs("docs/images", exist_ok=True)
    
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--force-device-scale-factor=1")
    
    driver = webdriver.Chrome(options=opts)
    driver.set_window_size(1920, 1080)
    
    url = "https://atharva110409-smart-factory-6g-network-appdashboard-p9hb2s.streamlit.app/"
    print(f"Navigating to {url}...")
    driver.get(url)
    
    # Wait for iframe containing Streamlit app
    print("Waiting for Streamlit iframe...")
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.TAG_NAME, "iframe"))
    )
    time.sleep(3)
    
    # Locate the app iframe (usually the first one with src containing 'streamlit.app')
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    app_iframe = None
    for f in iframes:
        src = f.get_attribute("src") or ""
        if "streamlit.app" in src:
            app_iframe = f
            break
    if not app_iframe and iframes:
        app_iframe = iframes[0]
        
    print(f"Switching to app iframe: {app_iframe.get_attribute('src')}")
    driver.switch_to.frame(app_iframe)
    
    # Wait for scorecards to appear
    print("Waiting for KPI scorecards to appear...")
    WebDriverWait(driver, 40).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Avg Network Latency') or contains(text(), 'Active Breakpoint')]"))
    )
    print("KPI scorecards found!")
    
    # Wait for Plotly charts to be rendered
    print("Waiting for Plotly charts (.js-plotly-plot or svg)...")
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".js-plotly-plot, svg.main-svg"))
        )
        print("Plotly chart detected!")
    except Exception as e:
        print(f"Warning waiting for plotly charts: {e}")
        
    # Wait additional seconds for all charts to finish rendering animations
    time.sleep(6)
    
    # Switch back to default content to capture full browser window
    driver.switch_to.default_content()
    overview_path = os.path.abspath("docs/images/dashboard_overview.png")
    driver.save_screenshot(overview_path)
    print(f"Saved Tab 1 screenshot to: {overview_path} (size: {os.path.getsize(overview_path)} bytes)")
    
    # Switch back to app iframe to click Tab 5
    driver.switch_to.frame(app_iframe)
    print("Finding tabs...")
    tabs = driver.find_elements(By.XPATH, "//button[@role='tab' or contains(@data-baseweb, 'tab')]")
    print(f"Found {len(tabs)} tabs:")
    # Find Tab 5: Network Scenario Simulator
    sim_tab = None
    for idx, t in enumerate(tabs):
        try:
            tab_text = t.text
        except Exception:
            tab_text = ""
        if "Simulator" in tab_text or "Scenario" in tab_text or "5." in tab_text or idx == 4:
            sim_tab = t
            break
            
    if sim_tab:
        print("Clicking Tab 5...")
        driver.execute_script("arguments[0].scrollIntoView();", sim_tab)
        sim_tab.click()
        
        # Wait for simulator to render
        print("Waiting for simulator content to render...")
        time.sleep(6)
        
        # Switch back to default content and screenshot
        driver.switch_to.default_content()
        sim_path = os.path.abspath("docs/images/dashboard_simulator.png")
        driver.save_screenshot(sim_path)
        print(f"Saved Tab 5 screenshot to: {sim_path} (size: {os.path.getsize(sim_path)} bytes)")
    else:
        print("Error: Could not find Simulator tab!")
        
    driver.quit()
    print("Screenshot capture complete!")

if __name__ == "__main__":
    capture_screenshots()
