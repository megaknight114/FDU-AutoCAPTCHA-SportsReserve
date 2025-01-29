# Fudan University Sports Reservation Script (with automatic CAPTCHA recognition) 复旦大学体育场馆自动预订脚本（含验证码识别）

Overview
--------

This script automates the reservation process for Fudan University's sports facilities, including automatic CAPTCHA recognition. It is designed to streamline the booking experience, making it faster and more efficient. By combining this script with time control tools (e.g., the "Task Scheduler" app in Windows), you can automate bookings without needing to wake up at 7 AM.

> **Disclaimer:**  
> 
> This script is provided for educational and personal use only. Use it responsibly and in compliance with Fudan University's terms and policies. The authors are not liable for any misuse or consequences arising from its usage.

## Requirements

* **Python 3**
* **ChromeDriver** (other drivers are supported, but you'll need to modify the relevant sections of the code)
* Necessary Python libraries (see `requirements.txt`)

> It is recommended to use the latest stable versions of all dependencies.

Installation
------------

1. Clone this repository:
   `git clone <repository_url>`

2. Install the required dependencies:
   `pip install -r requirements.txt`

## Known Issues

1. **CAPTCHA Click Bias**: Pyautogui may fail to click accurately on the CAPTCHA due to coordinate misalignment. If this occurs, use the `coordinate_detection.py` script to locate the top-left corner of the CAPTCHA image and update the values in `config.py`.

2. **Network Dependency**: The script's robustness is limited. Unexpected behavior may occur if the network connection is poor.

3. **CAPTCHA Recognition Accuracy**:  
   The CAPTCHA recognition algorithm combines clustering techniques and a MobileNet neural network. While generally effective, it may fail with an estimated 10% error rate. If this happens, the console will typically display the error: `'list index out of range'`.

Usage
-----

1. Complete the personalized setup in `main.py` and `config.py`. Detailed guidance is provided within the files.
   
   > **Important:** If necessary, modify the unannotated sections carefully. The following are the critical settings that should be updated:
   
   * **`main.py`:**
     * `username`
     * `password`
     * `target_campus`
     * `list_order`
     * `next_week`
     * `target_day`
     * `target_time_period`
   * **`config.py`:**
     * `x_offset`
     * `y_offset`
     * `driver_location`

2. Run `main.py` to start the script.

* * *

Contributing
------------

We welcome contributions from the community!  

If you have suggestions for improvements, bug fixes, or new features, feel free to fork this repository and submit a pull request. Please ensure that your code is well-documented and tested before submission.
