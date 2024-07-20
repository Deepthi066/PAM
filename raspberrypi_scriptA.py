import time
import board
import adafruit_dht
import adafruit_bno055

# Initialize the DHT22 sensor
dhtDevice = adafruit_dht.DHT22(board.D4)

# Initialize the BNO055 gyro sensor
i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)

# Function to convert Celsius to Kelvin
def celsius_to_kelvin(celsius):
    return celsius + 273.15

while True:
    try:
        # Read temperature from DHT22 sensor
        temperature_c = dhtDevice.temperature
        temperature_k = celsius_to_kelvin(temperature_c)
        
        # Read gyro data from BNO055 sensor
        gyro_data = sensor.gyro
        
        # Print sensor values
        print(f"Temperature (Kelvin): {temperature_k:.2f}")
        print(f"Gyro Data (X, Y, Z): {gyro_data}")
        
        # Wait for 5 seconds before next reading
        time.sleep(5)
        
    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print(error.args[0])
        time.sleep(2.0)
        continue
    except Exception as error:
        dhtDevice.exit()
        raise error