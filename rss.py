import subprocess
import re

def get_wifi_signals():
    try:
        result = subprocess.check_output(["netsh", "wlan", "show", "network", "mode=Bssid"], universal_newlines=True)
        networks = []
        ssid = None
        
        for line in result.split("\n"):
            if "SSID" in line:
                ssid = line.split(":")[1].strip()
            if "Signal" in line:
                signal = re.search(r"(\d+)%", line).group(1)
                networks.append((ssid, signal))
        
        return [networks, result]
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    wifi_signals = get_wifi_signals()
    print("initn========================")
    print(wifi_signals)
    # print(results)
    for ssid, signal in wifi_signals[0]:
        print(f"SSID: {ssid}, Signal: {signal}%")
