import urllib.request
import urllib.error
import os

# Test URLs
model_url = "https://github.com/ParshvaDongare/solar-forecasting-app/releases/download/v1.0.0/final_solar_forecasting_model_new_features.pkl"
data_url = "https://github.com/ParshvaDongare/solar-forecasting-app/releases/download/v1.0.0/final_combined_Data_CI.csv"

def test_download(url, filename):
    print(f"Testing download of {filename}...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/octet-stream'
        }
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            content_type = response.headers.get('Content-Type', '')
            print(f"Content-Type: {content_type}")
            
            # Check content length
            content_length = response.headers.get('Content-Length')
            if content_length:
                print(f"Content-Length: {content_length} bytes")
            
            # Read first 100 bytes to check if it's HTML
            data = response.read(100)
            if b'html' in data.lower() or b'<html' in data.lower():
                print("❌ Got HTML content instead of file!")
                return False
                
            # Download full file
            print(f"Downloading {filename}...")
            with open(filename, 'wb') as f:
                f.write(data)  # Write first 100 bytes
                # Continue reading the rest
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"✅ Downloaded {filename} ({size} bytes)")
                return True
            else:
                print("❌ File not created")
                return False
                
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e.code} {e.reason}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test downloads
test_download(model_url, "test_model.pkl")
test_download(data_url, "test_data.csv")