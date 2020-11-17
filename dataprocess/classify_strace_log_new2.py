# 把原有数据集使用selenium框架去KOODOUS查找相关信息,对数据集进行分类,分为正常和异常

import selenium.webdriver.support.ui as ui
from selenium import webdriver


def classify_log():
    file1 = open("H:\\A数据集\\others\\Dynamic_features\\Dynamic_features\\MARKOV_CHAINS_TRANSITIONS.csv", mode='r')
    file2 = open("F:\\pycharmproject\\GraduationProject\\data\\特征数据集.csv", mode='a')
    # 打开浏览器
    # 设置本地代理
    # chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--proxy-server=http://127.0.0.1:1081')
    browser = webdriver.Chrome()
    wait = ui.WebDriverWait(browser, 1.5)
    try:
        browser.get('https://koodous.com/apks')
        for line in file1.readlines():
            num1 = line.find(",")
            # print(file1)
            browser.find_element_by_xpath("/html/body/div/section/div/ng-include/div/div/div[1]/form/div[1]/input") \
                .clear()
            browser.find_element_by_xpath("/html/body/div/section/div/ng-include/div/div/div[1]/form/div[1]/input") \
                .send_keys(line[0:num1])
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
                continue
            if attribute:
                # 异常
                file2.write('-1,'+line[num1+1:])
            else:
                # 正常
                file2.write('1,' + line[num1+1:])
    except Exception as e:
        print(e)
        browser.close()


if __name__ == "__main__":
    classify_log()
