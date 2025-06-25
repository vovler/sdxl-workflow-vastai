import os
import time

def get_public_ip_and_port():
    public_ip = os.getenv("PUBLIC_IPADDR", "Not Set")
    port = os.getenv("VAST_TCP_PORT_80", "Not Set")
    return public_ip, port

def main():
    while True:
        public_ip, port = get_public_ip_and_port()
        print(f"My Public IP is: {public_ip} and my port is: {port}")
        time.sleep(5)

if __name__ == "__main__":
    main()
