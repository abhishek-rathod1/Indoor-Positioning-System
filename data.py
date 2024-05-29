import pywifi
from pywifi import const
import time
import pandas as pd
def scan_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(2)  # Allow some time for the scan to complete
    scan_results = iface.scan_results()

    networks = []
    for network in scan_results:
        ssid = network.ssid
        if ssid in target_ssids:
          signal = network.signal
          networks.append((ssid, signal))
    
    return networks

if __name__ == "__main__":
    target_ssids = ["Wifi_1", "Wifi_2", "Wifi_3", "Wifi_4"]
    Wifi_1 = []
    Wifi_2 = []
    Wifi_3 = []
    Wifi_4 = []
    # shooter = []
    # aykadam = []
    # abhira = []
    steps = []
    while True:
        step = input("Enter step value: ")
        if step == "N":
            break
        wifi_signals = scan_wifi()
        print(type(wifi_signals[0]))
    # for ssid, signal in wifi_signals:
    #     print(f"SSID: {ssid}, Signal: {signal} dBm")
        rss_values = {}
        for network in wifi_signals:
            rss_values[network[0]] = network[1]
        print(rss_values)
        rss_value_list = []
        for value in rss_values:
            rss_value_list.append(rss_values[value])
        print(rss_value_list)
        Wifi_1.append(rss_values["Wifi_1"])
        Wifi_2.append(rss_values["Wifi_2"])
        Wifi_3.append(rss_values["Wifi_3"])
        Wifi_4.append(rss_values["Wifi_4"])
        # shooter.append(rss_values["Shooter"])
        # aykadam.append(rss_values["aykadam"])
        # abhira.append(rss_values["Abhira"])
        steps.append(step)
    df = pd.DataFrame({
        "step": steps,
        "Wifi_1": Wifi_1,
        "Wifi_2": Wifi_2,
        "Wifi_3": Wifi_3,
        "Wifi_4": Wifi_4,
    })
    # df = pd.DataFrame({
    #     "step": steps,
    #     "shooter": shooter,
    #     "aykadam": aykadam,
    #     "abhira": abhira
    # })
    df.to_csv("data.csv", index=False)


