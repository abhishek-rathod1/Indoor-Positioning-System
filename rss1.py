import pywifi
from pywifi import const
import time

def scan_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(2)  # Allow some time for the scan to complete
    scan_results = iface.scan_results()

    networks = []
    for network in scan_results:
        ssid = network.ssid
        signal = network.signal
        networks.append((ssid, signal))
    
    return networks

if __name__ == "__main__":
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
