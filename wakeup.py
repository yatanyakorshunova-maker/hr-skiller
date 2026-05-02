from playwright.sync_api import sync_playwright
import time

URL = "https://hr-skiller-4fna6myyn4ja2ftgbcob3y.streamlit.app"

with sync_playwright() as p:
    print(f"Запускаем браузер для {URL}")
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    # Идём на страницу
    page.goto(URL, timeout=60000)
    
    # Ждём 10 секунд — даём время React/Streamlit загрузиться
    time.sleep(10)
    
    # Пробуем нажать кнопку "Yes, get this app back up!" если она есть
    try:
        wake_button = page.get_by_role("button", name="Yes, get this app back up!")
        if wake_button.count() > 0:
            wake_button.click()
            print("Нажата кнопка пробуждения")
            time.sleep(5)
    except:
        pass  # Кнопки нет — приложение уже активно
    
    print("Готово!")
    browser.close()
