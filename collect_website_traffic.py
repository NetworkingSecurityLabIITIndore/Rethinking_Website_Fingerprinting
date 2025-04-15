import os
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

# Configuration
tor_path = "path_to_tor_browser_folder/tor-browser-linux-x86_64-13.0.16/tor-browser/firefox"  # Path to the Tor Browser executable
websites = ["Enter_the_website_links_here"]  # List of websites to visit
capture_dir = "save_pcap"  # Directory to save pcap files
tcpdump_interface = "interface_name"  # Network interface to capture packets from

# Enter number of time you visit the website
n = int(input("Enter number of times you want to visit the website : "))

# Ensure the capture directory exists
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

# Set up Tor Browser to always connect (add this to the torrc file)
torrc_path = "path_to_tor_browser_folder/tor-browser-linux-x86_64-13.0.16/tor-browser/Data/Tor/torrc"
with open(torrc_path, 'a') as torrc:
    torrc.write("\nDisableNetwork 0\n")  # Ensure Tor Browser connects automatically

# Set up a custom Firefox profile for Tor Browser
profile_path = "path_to_tor_browser_folder/tor-browser-linux-x86_64-13.0.16/tor-browser/Data/Browser/profile.default"
profile = webdriver.FirefoxProfile(profile_path)

# Modify the profile to bypass the initial network settings dialog
profile.set_preference("extensions.torlauncher.prompt_at_startup", False)
profile.set_preference("network.proxy.type", 1)  # Use system proxy settings (configured by Tor)

# Set up Selenium with Tor Browser
options = Options()
options.binary_location = tor_path
options.profile = profile

# Function to start tcpdump
def start_tcpdump(website, iteration):
    sanitized_name = website.replace('https://', '').replace('http://', '').replace('/', '_').replace('.', '_')
    pcap_file = os.path.join(capture_dir, f"{sanitized_name}_{iteration}.pcap")
    cmd = ["tcpdump", "-i", tcpdump_interface, "-w", pcap_file]
    return subprocess.Popen(cmd)


# Function to stop tcpdump
def stop_tcpdump(proc):
    proc.terminate()
    time.sleep(2)  # Give it some time to terminate gracefully
    if proc.poll() is None:  # Check if process is still running
        proc.kill()  # Force kill if it's still running
    proc.wait()

# Function to check if Tor has connected
def check_tor_connection():
    print("Waiting for Tor to connect...")
    time.sleep(10)  # Give some time for Tor to establish the connection
    # You can add more sophisticated checks here to ensure Tor is fully connected

# Visit websites and capture traffic
for i in range(n):
    for website in websites:  # Visit each website
        print(f"Iteration {i+1}, loading website: {website}")
        driver = None  # Initialize driver to None at the start of each loop
        tcpdump_proc = None  # Initialize tcpdump_proc to None

        try:
            # Start tcpdump to capture traffic on a specific interface
            tcpdump_proc = start_tcpdump(website, i+1)

            # Start the Tor Browser with Selenium
            driver = webdriver.Firefox(options=options)

            # Check if Tor is connected before proceeding
            check_tor_connection()

            # Load website
            driver.get(website)

            # Capture traffic for 30 seconds
            print("Capturing traffic for 30 seconds")
            time.sleep(30)

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Close the browser after each iteration if driver was initialized
            if driver:
                print("Closing the browser")
                driver.quit()

            # Stop tcpdump if it was started
            if tcpdump_proc:
                print("Stopping tcpdump")
                stop_tcpdump(tcpdump_proc)

            # Buffer sleep time after browser and tcpdump are closed
            print("Sleeping for 10 seconds before the next iteration")
            time.sleep(10)  # Add a delay between iterations
