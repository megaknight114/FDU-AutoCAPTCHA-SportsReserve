import pyautogui
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import base64
import cv2
import os
import random
import config
import utils_for_captcha
import sys
from config import *
options = webdriver.ChromeOptions()
options.add_argument("--force-device-scale-factor=1")
options.add_argument("--high-dpi-support=1")
wd = webdriver.Chrome(service=Service(r'C:\chromedriver.exe'), options=options)
link = "https://elife.fudan.edu.cn/public/front/myServices.htm?id=2c9c486e4f821a19014f82381feb0001"
def retry_click(location, location_before, retries=config.retries,before=1,interval=config.interval):
    attempt = 0
    while attempt < retries:
        try:
            print(f"Try to click {location} for the {attempt + 1} time")
            element = wd.find_element(By.XPATH, location)
            element.click()
            return
        except Exception:
            print(f"Element {location} not found, retrying...")
            attempt += 1
            if attempt < retries:
                try:
                    if before==1:
                        element = wd.find_element(By.XPATH, location_before)
                        element.click()
                    time.sleep(interval)
                except Exception:
                    print(f"Fallback element {location_before} also not found.")
                    time.sleep(interval)
                    continue
            else:
                sys.exit(f"Function 'retry_click' failed after {retries} retries")

def refresh_substitute(days, days_refresh):
    retry_click(days_refresh,'default',before=0)
    retry_click(days, 'default', before=0)

def login(username, password, auto=1):
    wd.get(link)
    wd.maximize_window()
    retry_click('//*[@id="login_table_div"]/div[2]/input','default',before=0)
    if auto==1:
        element=wd.find_element(By.XPATH, '//*[@id="username"]')
        element.send_keys(username)
        element=wd.find_element(By.XPATH, '//*[@id="password"]')
        element.send_keys(password)
        retry_click('//*[@id="idcheckloginbtn"]', 'default', before=0)
        print("Login Successful!")
    else:
        time.sleep(config.login_time)
def prepare(area, place, week):
    wd.switch_to.window(wd.window_handles[-1])
    wd.switch_to.frame('contentIframe')
    retry_click(area, 'default', before=0)
    retry_click(place, 'default', before=0)
    print("Location identified!")
    if week == 1:
        retry_click('/html/body/div[2]/div[2]/div[1]/ul/li[10]', 'default', before=0)
        print("Next page!")

def get_img(timex,days):
    retry_click(timex, days)
    retry_click('//*[@id="verify_button1"]',timex)

def verify_image():
    image_element = wd.find_element(By.CSS_SELECTOR,
                                    'body > div.valid_popup > div.valid_modal > div.valid_modal__body > div > div.valid_panel > div.valid_bgimg > img')
    current_dir = os.path.dirname(__file__)
    save_folder = os.path.join(current_dir, "save_folder")
    if image_element:
        i = 0
        image_url = None
        while image_url is None:
            if i < 15:
                image_url = image_element.get_attribute('src')
                time.sleep(0.01)
                i += 1
            if i == 15:
                element = wd.find_element(By.XPATH, '/html/body/div[7]/div[2]/div[2]/div/div[1]/div[3]/button')
                element.click()
                i = 0
    else:
        print("Failed to find the image")
        sys.exit("The process has ended, exit code 1")
    save_path = os.path.join(save_folder, 'image.jpg')
    try:
        header, base64_data = image_url.split(',', 1)
    except:
        element = wd.find_element(By.XPATH, '/html/body/div[7]/div[2]/div[2]/div/div[1]/div[3]/button')
        element.click()
        image_url = image_element.get_attribute('src')
        header, base64_data = image_url.split(',', 1)
    image_data = base64.b64decode(base64_data)
    with open(save_path, 'wb') as file:
        file.write(image_data)
        print(f"The image has been downloaded to {save_path} successfully")
    image_path = save_path
    image = cv2.imread(image_path)
    word, score, segment = utils_for_captcha.match_word(image)
    print(word, score, segment)
    image_element = wd.find_element(By.CSS_SELECTOR,
                                    'body > div.valid_popup > div.valid_modal > div.valid_modal__body > div > div.valid_panel > div.valid_bgimg > img')
    size = image_element.size
    refined_centers = []
    width, height = size["width"], size["height"]
    for i in range(4):
        refined_x = utils_for_captcha.global_centers[i][1] / 800 * width
        refined_y = utils_for_captcha.global_centers[i][0] / 600 * height
        refined_centers.append([refined_x, refined_y])
    sorted_centers = sorted(refined_centers, key=lambda x: x[0])
    for i in range(4):
        index = segment[i] - 1
        x_offset = sorted_centers[index][0]
        y_offset = sorted_centers[index][1]
        wd.execute_script("arguments[0].scrollIntoView(true);", image_element)
        pyautogui.moveTo(x_offset + config.x_offset, y_offset + config.y_offset, duration=config.mouse_move_speed)
        pyautogui.click()
        time.sleep(random.randint(50, 100) / 50000)
    print("CAPTCHA clicked!")

def book():
    element = wd.find_element(By.XPATH, '//*[@id="btn_sub"]')
    element.click()
    try:
        time.sleep(1)
        wd.find_element(By.XPATH, '//*[@id="orderCommit"]/div/div/div[2]/div[2]/table/tbody/tr[6]/td[1]')
        print('Booked successfully!')
    except:
        time.sleep(0.5)
        pyautogui.moveTo(1056, 181, duration=config.mouse_move_speed)
        pyautogui.click()
        element = wd.find_element(By.XPATH, '//*[@id="btn_sub"]')
        element.click()
        time.sleep(1)
        wd.find_element(By.XPATH, '//*[@id="orderCommit"]/div/div/div[2]/div[2]/table/tbody/tr[6]/td[1]')
        print('Booked successfully!')

