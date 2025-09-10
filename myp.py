import os
import time
import re
import base64
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import numpy as np
import pytesseract
from io import BytesIO

class EbookDownloader:
    def __init__(self, headless=True):
        self.options = webdriver.ChromeOptions()
        if headless:
            self.options.add_argument('--headless')
        self.options.add_argument('--start-maximized')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--disable-infobars')
        self.options.add_argument('--disable-extensions')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--disable-blink-features=AutomationControlled')
        self.driver = None
        self.screenshot_count = 0
        self.temp_dir = "ebook_screenshots"
        
    def init_driver(self):
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.options
        )
        # 创建临时目录
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def login(self, username, password):
        """登录京东账号（如果需要）"""
        login_url = "https://passport.jd.com/new/login.aspx"
        self.driver.get(login_url)
        time.sleep(2)
        
        # 切换到账号登录
        account_tab = self.driver.find_element(By.CSS_SELECTOR, "div.login-tab.login-tab-r")
        account_tab.click()
        time.sleep(1)
        
        # 输入用户名
        username_input = self.driver.find_element(By.ID, "loginname")
        username_input.send_keys(username)
        
        # 输入密码
        password_input = self.driver.find_element(By.ID, "nloginpwd")
        password_input.send_keys(password)
        
        # 提交登录
        submit_btn = self.driver.find_element(By.ID, "loginsubmit")
        submit_btn.click()
        
        # 等待登录完成
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "nickname"))
        )
        print("登录成功！")
    
    def open_ebook(self, url):
        """打开电子书页面"""
        self.driver.get(url)
        print("正在加载电子书页面...")
        
        # 等待主要内容加载
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".reader_main"))
        )
        
        # 关闭干扰元素
        self.remove_annoying_elements()
        
        print("电子书页面加载完成")
    
    def remove_annoying_elements(self):
        """移除页面上的干扰元素"""
        js = """
        // 移除顶部栏
        var topbar = document.querySelector('.read-topbar');
        if(topbar) topbar.style.display = 'none';
        
        // 移除底部栏
        var bottombar = document.querySelector('.read-bottombar');
        if(bottombar) bottombar.style.display = 'none';
        
        // 移除广告
        var ads = document.querySelectorAll('.ad-container, .ad-banner');
        ads.forEach(function(ad) {
            ad.style.display = 'none';
        });
        
        // 移除购买提示
        var purchasePrompts = document.querySelectorAll('.purchase-prompt, .buy-prompt');
        purchasePrompts.forEach(function(prompt) {
            prompt.style.display = 'none';
        });
        """
        self.driver.execute_script(js)
    
    def capture_full_book(self):
        """捕获整本电子书"""
        print("开始捕获电子书内容...")
        
        # 获取页面总高度
        total_height = self.driver.execute_script("return document.body.scrollHeight")
        viewport_height = self.driver.execute_script("return window.innerHeight")
        scrolls = int(np.ceil(total_height / viewport_height))
        
        print(f"预计需要滚动 {scrolls} 次")
        
        # 创建屏幕截图
        screenshots = []
        for i in range(scrolls):
            # 滚动到当前位置
            scroll_pos = i * viewport_height
            self.driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
            
            # 等待内容稳定
            time.sleep(1.5)  # 根据实际页面加载速度调整
            
            # 截取当前视口
            screenshot_path = os.path.join(self.temp_dir, f"screenshot_{self.screenshot_count}.png")
            self.driver.save_screenshot(screenshot_path)
            screenshots.append(screenshot_path)
            self.screenshot_count += 1
            
            print(f"已捕获 {i+1}/{scrolls} 屏 ({scroll_pos}/{total_height}px)")
            
            # 检查是否有"购买提示"出现
            if self.check_purchase_prompt():
                print("检测到购买提示，无法继续捕获")
                break
        
        print("内容捕获完成")
        return screenshots
    
    def check_purchase_prompt(self):
        """检查页面是否有购买提示"""
        try:
            prompt = self.driver.find_element(By.CSS_SELECTOR, ".purchase-prompt, .buy-prompt")
            if prompt.is_displayed():
                return True
        except:
            pass
        return False
    
    def stitch_screenshots(self, screenshots, output_pdf):
        """将截图拼接成PDF"""
        print("开始拼接截图...")
        
        images = []
        for screenshot in screenshots:
            img = Image.open(screenshot)
            images.append(img.convert('RGB'))
        
        # 保存为PDF
        images[0].save(
            output_pdf,
            save_all=True,
            append_images=images[1:],
            resolution=100.0
        )
        
        print(f"电子书已保存为: {output_pdf}")
    
    def cleanup(self):
        """清理临时文件"""
        if self.driver:
            self.driver.quit()
        
        # 删除临时截图
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        print("临时文件已清理")

if __name__ == "__main__":
    # 配置参数
    EBOOK_URL = "https://cread.jd.com/read/startRead.action?bookId=30310395&readType=1"  # 替换为实际URL
    OUTPUT_NAME = "Android开发艺术探索"  # 输出文件名
    USERNAME = "peiyuan688"  # 如果需要登录
    PASSWORD = "jingdong1999"  # 如果需要登录
    
    downloader = EbookDownloader(headless=False)  # 设为False以便调试
    try:
        downloader.init_driver()
        
        # 如果需要登录
        if USERNAME and PASSWORD:
            downloader.login(USERNAME, PASSWORD)
        
        # 打开电子书
        downloader.open_ebook(EBOOK_URL)
        
        # 捕获电子书
        screenshots = downloader.capture_full_book()
        
        # 拼接为PDF
        downloader.stitch_screenshots(screenshots, f"{OUTPUT_NAME}.pdf")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        downloader.cleanup()