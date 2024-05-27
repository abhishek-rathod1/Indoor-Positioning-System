import time
import openpyxl
from pywifi import PyWiFi

# Function to scan for Wi-Fi networks and collect RSSI data
def scan_wifi_rssi(interface_name="wlan0"):
    wifi = PyWiFi()
    iface = wifi.interfaces()[0]  # Adjust the index if you have multiple interfaces

    iface.scan()
    time.sleep(2)
    scan_results = iface.scan_results()

    wifi_data = []
    for result in scan_results:
        ssid = result.ssid
        rssi = result.signal
        wifi_data.append((ssid, rssi))

    return wifi_data

# Function to save data to an Excel file with location input
def save_to_excel(data, excel_filename="wifi_rssi_data9.xlsx"):
    try:
        wb = openpyxl.load_workbook(excel_filename, read_only=False, data_only=True)
        sheet = wb.active
    except FileNotFoundError:
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.append(["SSID", "RSSI", "Location", "Timestamp"])

    location = input("Enter the physical location: ")

    for ssid, rssi in data:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        sheet.append([ssid, rssi, location, timestamp])

    wb.save(excel_filename)
    wb.close()

# Main loop to continuously capture and save data
while True:
    wifi_data = scan_wifi_rssi()
    save_to_excel(wifi_data)
    time.sleep(60)  # Adjust the sleep interval as needed
