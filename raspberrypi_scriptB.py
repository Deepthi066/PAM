import time
import Adafruit_DHT  # Library for DHT22 sensor
import smbus  # Library for I2C communication
from math import isnan

# Function to read temperature from DHT22 sensor
def read_temperature():
    sensor = Adafruit_DHT.DHT22
    pin = 4  # GPIO pin number where the DHT22 sensor is connected
    humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
    if not isnan(temperature):
        return temperature
    else:
        return None

# Function to read rotational speed from gyro sensor
def read_rotational_speed():
    bus = smbus.SMBus(1)  # Use '1' for Raspberry Pi revision 2 (Model B+)
    address = 0x6b  # I2C address of the gyro sensor
    # Write to control register to turn on gyro
    bus.write_byte_data(address, 0x20, 0x0F)
    time.sleep(0.2)
    # Read gyro values
    data = bus.read_i2c_block_data(address, 0x28 | 0x80, 6)
    gyro_x = data[0] | data[1] << 8
    gyro_y = data[2] | data[3] << 8
    gyro_z = data[4] | data[5] << 8
    # Convert raw values to degrees per second
    gyro_x = (gyro_x * 0.07)  # Sensitivity scale 70 mdps/digit
    gyro_y = (gyro_y * 0.07)
    gyro_z = (gyro_z * 0.07)
    return gyro_x, gyro_y, gyro_z

# Function to read tool wear from Vibrationsw420 sensor
def read_tool_wear():
    # Simulated function for reading tool wear from Vibrationsw420 sensor
    return 0  # Replace with actual function to read tool wear

# Main function to continuously read sensor values and print them
def main():
    while True:
        # Read temperature from DHT22 sensor
        air_temperature = read_temperature()
        if air_temperature is not None:
            process_temperature = air_temperature + 10  # Process temperature is air temperature + 10 K
        else:
            print("Failed to read temperature from DHT22 sensor")
            process_temperature = None

        # Read rotational speed from gyro sensor
        gyro_x, gyro_y, gyro_z = read_rotational_speed()

        # Read tool wear from Vibrationsw420 sensor
        tool_wear = read_tool_wear()

        # Print sensor values
        print(f"Air temperature: {air_temperature:.2f} K")
        print(f"Process temperature: {process_temperature:.2f} K")
        print(f"Rotational speed (X-axis): {gyro_x:.2f} dps")
        print(f"Rotational speed (Y-axis): {gyro_y:.2f} dps")
        print(f"Rotational speed (Z-axis): {gyro_z:.2f} dps")
        print(f"Tool wear: {tool_wear:.2f} min")
        print("-" * 20)

        # Wait for 5 seconds before reading values again
        time.sleep(5)

if __name__ == "__main__":
    main()
