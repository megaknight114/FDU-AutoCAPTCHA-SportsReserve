import autooperating_module
username = ''  # UIS login student ID
password = ''  # UIS login password
auto_login = 1  # 1 for auto-login, 0 for manual login
target_campus = 5  # Target campus: 2 for Handan, 3 for Fenglin, 4 for Zhangjiang, 5 for Jiangwan
list_order = 1  # Target venue: nth venue from top to bottom on the selected campus page (e.g., 1 for Jiangwan badminton courts)
next_week = 0  # 0 for this week (Monday to Sunday), 1 for next week
target_day = 3  # Target day of the week (1 for Monday, 2 for Tuesday, ..., 7 for Sunday)
target_time_period = 10  # nth time slot from top to bottom
area=f'//*[@id="queryForm"]/table[2]/tbody/tr[1]/td[2]/a[{target_campus}]'
place=f'/html/body/div/div/div[3]/table[{list_order}]/tbody/tr/td[2]/table/tbody/tr[5]/td/a'
days=f'//*[@id="one{target_day}"]'
days_refresh=f'//*[@id="one{target_day%7+1}"]'
timex=f'//*[@id="con_one_1"]/table/tbody[{target_time_period}]/tr/td[6]/img'
autooperating_module.login(username=username, password=password,auto=auto_login)
autooperating_module.prepare(area=area, place=place,week=next_week)
autooperating_module.refresh_substitute(days=days,days_refresh=days_refresh)
autooperating_module.get_img(timex=timex,days=days)
autooperating_module.verify_image()
autooperating_module.book()