import os
import pandas as pd
from scapy.all import rdpcap
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, entropy
from collections import Counter

# Define path to your pcap files
pcap_path = "PCAP_folder_path"
pcap_folder = os.path.expanduser(pcap_path)

# Define IP addresses of interest
ips = ['Enter_IP_address_on_which_data_collected']
web_name = "Enter_website_name_here"


def count_packets(pcap_file):
    """Count the number of packets in a pcap file using scapy."""
    packets = rdpcap(pcap_file)
    return len(packets)

def calculate_statistics(sizes):
    """Calculate average and standard deviation for a list of packet sizes."""
    if sizes:
        avg_size = round(np.mean(sizes), 10)
        std_dev = round(np.std(sizes), 10)
    else:
        avg_size = 0
        std_dev = 0
    return avg_size, std_dev

def calculate_transmission_time_stats(timestamps):
    """Calculate statistics for transmission times."""
    if len(timestamps) > 1:
        timestamps = np.array(timestamps, dtype=float)
        transmission_times = np.diff(timestamps)
        variance_time = round(np.var(transmission_times), 10)
        std_dev_time = round(np.std(transmission_times), 10)
        min_time = round(np.min(transmission_times), 10)
        max_time = round(np.max(transmission_times), 10)
        median_time = round(np.median(transmission_times), 10)
        skewness_time = round(skew(transmission_times), 10)
    else:
        variance_time = 0
        std_dev_time = 0
        min_time = 0
        max_time = 0
        median_time = 0
        skewness_time = 0
    
    return variance_time, std_dev_time, min_time, max_time, median_time, skewness_time

def calculate_inter_arrival_time_stats(timestamps):
    """Calculate statistics for inter-arrival times."""
    if len(timestamps) > 1:
        timestamps = np.array(timestamps, dtype=float)
        inter_arrival_times = np.diff(timestamps)
        mean_inter_arrival = round(np.mean(inter_arrival_times), 10)
        variance_inter_arrival = round(np.var(inter_arrival_times), 10)
        std_dev_inter_arrival = round(np.std(inter_arrival_times), 10)
        min_inter_arrival = round(np.min(inter_arrival_times), 10)
        median_inter_arrival = round(np.median(inter_arrival_times), 10)
        skewness_inter_arrival = round(skew(inter_arrival_times), 10)
    else:
        mean_inter_arrival = 0
        variance_inter_arrival = 0
        std_dev_inter_arrival = 0
        min_inter_arrival = 0
        median_inter_arrival = 0
        skewness_inter_arrival = 0
    
    return mean_inter_arrival, variance_inter_arrival, std_dev_inter_arrival, min_inter_arrival, median_inter_arrival, skewness_inter_arrival

def detect_bursts(timestamps, burst_threshold=0.1, min_burst_size=2):
    """Detect bursts in packet timestamps."""
    bursts = []
    current_burst = []
    
    for i in range(1, len(timestamps)):
        time_diff = timestamps[i] - timestamps[i - 1]
        if time_diff <= burst_threshold:
            current_burst.append(timestamps[i])
        else:
            if len(current_burst) >= min_burst_size:
                bursts.append(current_burst)
            current_burst = [timestamps[i]]
    
    if len(current_burst) >= min_burst_size:
        bursts.append(current_burst)
    
    burst_durations = [burst[-1] - burst[0] for burst in bursts]
    burst_frequency = len(bursts) / ((timestamps[-1] - timestamps[0]) / 60)  # bursts per minute
    
    return burst_durations, burst_frequency

def calculate_entropy(sizes):
    """Calculate entropy of packet sizes."""
    if sizes:
        size_counts = Counter(sizes)
        size_probabilities = [count / len(sizes) for count in size_counts.values()]
        ent = entropy(size_probabilities, base=2)
        return round(ent, 10)
    return 0

def count_incoming_outgoing_packets(pcap_file, ips):
    """Count incoming and outgoing packets based on IP addresses and their sizes, and calculate transmission times."""
    packets = rdpcap(pcap_file)
    incoming_count = 0
    outgoing_count = 0
    total_size = 0
    total_incoming_size = 0
    total_outgoing_size = 0
    incoming_sizes = []
    outgoing_sizes = []
    timestamps = []
    packet_lengths = []
    
    for packet in packets:
        if packet.haslayer('IP'):
            ip_layer = packet.getlayer('IP')
            packet_size = len(packet)
            total_size += packet_size
            timestamps.append(float(packet.time))  # Ensure timestamp is a float
            packet_lengths.append(packet_size)
            if ip_layer.src in ips:
                outgoing_count += 1
                total_outgoing_size += packet_size
                outgoing_sizes.append(packet_size)
            elif ip_layer.dst in ips:
                incoming_count += 1
                total_incoming_size += packet_size
                incoming_sizes.append(packet_size)

    
    avg_outgoing_size, std_dev_outgoing_size = calculate_statistics(outgoing_sizes)
    avg_incoming_size, std_dev_incoming_size = calculate_statistics(incoming_sizes)
    
    if timestamps:
        total_transmission_time = round(max(timestamps) - min(timestamps), 10)
        avg_transmission_time = round(total_transmission_time / len(timestamps), 10)
        variance_transmission_time, std_dev_transmission_time, min_transmission_time, max_transmission_time, median_transmission_time, skewness_transmission_time = calculate_transmission_time_stats(timestamps)
        rate_of_transmission = round(len(timestamps) / total_transmission_time, 10) if total_transmission_time > 0 else 0
        rate_of_packet_arrival = round(len(timestamps) / total_transmission_time, 10) if total_transmission_time > 0 else 0
        burst_durations, burst_frequency = detect_bursts(timestamps)
        avg_burst_duration = round(np.mean(burst_durations), 10) if burst_durations else 0
    else:
        total_transmission_time = 0
        avg_transmission_time = 0
        variance_transmission_time = 0
        std_dev_transmission_time = 0
        min_transmission_time = 0
        max_transmission_time = 0
        median_transmission_time = 0
        skewness_transmission_time = 0
        rate_of_transmission = 0
        rate_of_packet_arrival = 0
        avg_burst_duration = 0
        burst_frequency = 0
    
    if packet_lengths:
        median_packet_length = round(np.median(packet_lengths), 10)
        average_bytes_per_packet = round(np.mean(packet_lengths), 10)
        
        # New features
        min_packet_size = round(np.min(packet_lengths), 10)
        max_packet_size = round(np.max(packet_lengths), 10)
        mode_size = round(Counter(packet_lengths).most_common(1)[0][0], 10)
        p75_size = round(np.percentile(packet_lengths, 75), 10)
        p25_size = round(np.percentile(packet_lengths, 25), 10)
    else:
        median_packet_length = 0
        average_bytes_per_packet = 0
        min_packet_size = 0
        max_packet_size = 0
        mode_size = 0
        p75_size = 0
        p25_size = 0
    
    mean_inter_arrival, variance_inter_arrival, std_dev_inter_arrival, min_inter_arrival, median_inter_arrival, skewness_inter_arrival = calculate_inter_arrival_time_stats(timestamps)
    
    entropy_of_sizes = calculate_entropy(packet_lengths)
    
    incoming_to_outgoing_packet_ratio = round(incoming_count / outgoing_count, 10) if outgoing_count > 0 else 0
    incoming_to_outgoing_size_ratio = round(total_incoming_size / total_outgoing_size, 10) if total_outgoing_size > 0 else 0
    
    # Round values to 10 decimal places
    avg_outgoing_size = round(avg_outgoing_size, 10)
    std_dev_outgoing_size = round(std_dev_outgoing_size, 10)
    avg_incoming_size = round(avg_incoming_size, 10)
    std_dev_incoming_size = round(std_dev_incoming_size, 10)
    total_size = round(total_size, 10)
    total_incoming_size = round(total_incoming_size, 10)
    total_outgoing_size = round(total_outgoing_size, 10)

    
    return (incoming_count, outgoing_count, total_size, total_incoming_size, total_outgoing_size,
            avg_outgoing_size, std_dev_outgoing_size, avg_incoming_size, std_dev_incoming_size,
            total_transmission_time, avg_transmission_time,
            variance_transmission_time, std_dev_transmission_time, min_transmission_time, max_transmission_time,
            median_transmission_time, skewness_transmission_time, rate_of_packet_arrival,
            median_packet_length,
            mean_inter_arrival, variance_inter_arrival, std_dev_inter_arrival,
            min_inter_arrival, median_inter_arrival, skewness_inter_arrival,
            avg_burst_duration, burst_frequency,
            entropy_of_sizes, incoming_to_outgoing_packet_ratio, incoming_to_outgoing_size_ratio, average_bytes_per_packet,
            min_packet_size, max_packet_size, mode_size, p75_size, p25_size)  # Add new features

def extract_features_from_pcap(file_path, ips):
    """Extract features from pcap file including packet counts, incoming/outgoing counts, and transmission times."""
    total_packets = count_packets(file_path)
    (incoming_count, outgoing_count, total_size, total_incoming_size, total_outgoing_size,
     avg_outgoing_size, std_dev_outgoing_size, avg_incoming_size, std_dev_incoming_size,
     total_transmission_time, avg_transmission_time,
     variance_transmission_time, std_dev_transmission_time, min_transmission_time, max_transmission_time,
     median_transmission_time, skewness_transmission_time, rate_of_packet_arrival,
     median_packet_length,
     mean_inter_arrival, variance_inter_arrival, std_dev_inter_arrival,
     min_inter_arrival, median_inter_arrival, skewness_inter_arrival,
     avg_burst_duration, burst_frequency,
     entropy_of_sizes, incoming_to_outgoing_packet_ratio, incoming_to_outgoing_size_ratio, average_bytes_per_packet,
     min_packet_size, max_packet_size, mode_size, p75_size, p25_size) = count_incoming_outgoing_packets(file_path, ips)
    
    incoming_fraction = round(incoming_count / total_packets, 10) if total_packets > 0 else 0
    outgoing_fraction = round(outgoing_count / total_packets, 10) if total_packets > 0 else 0
    average_packet_size = round(total_size / total_packets, 10) if total_packets > 0 else 0
    
    Website_name = web_name
    
    # Initialize feature variables
    feature_values = [
        total_packets, 
        incoming_count, 
        outgoing_count,
        incoming_fraction,
        outgoing_fraction,
        total_size,
        average_packet_size,
        total_incoming_size,
        total_outgoing_size,
        avg_outgoing_size,
        std_dev_outgoing_size,
        avg_incoming_size,
        std_dev_incoming_size,
        total_transmission_time,
        avg_transmission_time,
        variance_transmission_time,
        std_dev_transmission_time,
        min_transmission_time,
        max_transmission_time,
        median_transmission_time,
        skewness_transmission_time,
        rate_of_packet_arrival,
        median_packet_length,
        mean_inter_arrival,
        variance_inter_arrival,
        std_dev_inter_arrival,
        min_inter_arrival,
        median_inter_arrival,
        skewness_inter_arrival,
        avg_burst_duration,
        burst_frequency,
        entropy_of_sizes,
        incoming_to_outgoing_packet_ratio,
        incoming_to_outgoing_size_ratio,
        average_bytes_per_packet,
        min_packet_size,  
        max_packet_size,  
        mode_size,        
        p75_size,        
        p25_size,         
        Website_name,
    ]
    
    return feature_values
    



columns = [
    'Total_number_of_packets', 
    'Incoming_packets', 
    'Outgoing_packets', 
    'Fraction_of_incoming_packets',
    'Fraction_of_outgoing_packets',
    'Total_transmission_size',
    'Average_packet_size',
    'Total_incoming_size',
    'Total_outgoing_size',
    'Avg_outgoing_packet_size',
    'Std_dev_outgoing_packet_size',
    'Avg_incoming_packet_size',
    'Std_dev_incoming_packet_size',
    'Total_transmission_time',
    'Avg_transmission_time',
    'Variance_transmission_time',
    'Std_dev_transmission_time',
    'Min_transmission_time',
    'Max_transmission_time',
    'Median_transmission_time',
    'Skewness_transmission_time',
    'Rate_of_packet_arrival',
    'Median_packet_length',
    'Mean_inter_arrival_time',
    'Variance_inter_arrival_time',
    'Std_dev_inter_arrival_time',
    'Min_inter_arrival_time',
    'Median_inter_arrival_time',
    'Skewness_inter_arrival_time',
    'Average_burst_duration',
    'Burst_frequency',
    'Entropy_of_packet_sizes',
    'Incoming_to_outgoing_packet_ratio',
    'Incoming_to_outgoing_size_ratio',
    'Average_bytes_per_packet',
    'Min_packet_size',  
    'Max_packet_size',  
    'Mode_of_packet_sizes',  
    '75th_Percentile_packet_size',   
    '25th_Percentile_packet_size',  
    'Website_name',
]

data = []

# Iterate over each pcap file and extract features
for pcap_file in tqdm(os.listdir(pcap_folder), desc="Processing pcap files"):
    if pcap_file.endswith('.pcap'):
        file_path = os.path.join(pcap_folder, pcap_file)
        features = extract_features_from_pcap(file_path, ips)
        data.append(features)

# Create and save the DataFrame
df = pd.DataFrame(data, columns=columns)
df.to_csv('feature.csv', index=False)

print("Feature extraction completed and CSV file saved as 'feature.csv'.")