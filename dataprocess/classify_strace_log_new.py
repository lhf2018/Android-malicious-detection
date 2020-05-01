# 使用selenium框架去KOODOUS查找相关信息,对数据集进行分类,分为正常和异常

from selenium import webdriver
import os
import time
import selenium.webdriver.support.ui as ui
import shutil


def move_normal_file(file1):
    shutil.move("H:\\A数据集\\Strace_OmniDroid_V2\\Strace\\" + file1
                , "H:\\A数据集\\normal\\" + file1)


def move_abnormal_file(file2):
    shutil.move("H:\\A数据集\\Strace_OmniDroid_V2\\Strace\\" + file2
                , "H:\\A数据集\\abnormal\\" + file2)


def move_no_file(file3):
    shutil.move("H:\\A数据集\\Strace_OmniDroid_V2\\Strace\\" + file3
                , "H:\\A数据集\\no\\" + file3)


def classify_log():
    path = "H:\\A数据集\\Strace_OmniDroid_V2\\Strace"
    files = os.listdir(path)
    # 打开浏览器
    # 设置本地代理
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--proxy-server=http://127.0.0.1:1081')
    browser = webdriver.Chrome(chrome_options=chrome_options)
    wait = ui.WebDriverWait(browser, 1.5)
    try:
        browser.get('https://koodous.com/apks')
        for file1 in files:
            num1 = file1.find(".")
            # print(file1)
            browser.find_element_by_xpath("/html/body/div/section/div/ng-include/div/div/div[1]/form/div[1]/input") \
                .clear()
            browser.find_element_by_xpath("/html/body/div/section/div/ng-include/div/div/div[1]/form/div[1]/input") \
                .send_keys(file1[0:num1])
            browser.find_element_by_xpath(
                "/html/body/div/section/div/ng-include/div/div/div[1]/form/div[1]/span/button") \
                .click()
            try:
                # 获取元素内容
                wait.until(
                    lambda driver: driver.find_element_by_xpath(
                        "/html/body/div/section/div/ng-include/div/div/div[3]/div/table/tbody/tr/td[1]/p[2]/span"))
                # print(file1)
                attribute = browser \
                    .find_element_by_xpath(
                    "/html/body/div/section/div/ng-include/div/div/div[3]/div/table/tbody/tr/td[4]/em") \
                    .is_displayed()
            except Exception:
                move_no_file(file1)
                continue
            if attribute:
                # 异常
                move_abnormal_file(file1)
            else:
                # 正常
                move_normal_file(file1)
    except Exception as e:
        print(e)
        browser.close()


if __name__ == "__main__":
    classify_log()
