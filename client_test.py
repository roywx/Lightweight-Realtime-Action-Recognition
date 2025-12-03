import socket
import json

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5000))

    print("Connected to server")

    buffer = ""
    while True:
        data = client.recv(1024).decode('utf-8')
        if not data:
            break
        
        buffer += data
        
        # Process complete messages (split by newline)
        while '\n' in buffer:
            message, buffer = buffer.split('\n', 1)
            if message:
                action_data = json.loads(message)
                
                lh = action_data['left_hand']
                rh = action_data['right_hand']
                print(f"Right Hand:  x={rh['x']:.4f}, y={rh['y']:.4f}, conf={rh['confidence']:.4f}")
                print(f"Left Hand:  x={lh['x']:.4f}, y={lh['y']:.4f}, conf={lh['confidence']:.4f}")
                print(f"Action: {action_data['action']}, Confidence: {action_data['confidence']}")


if __name__ == "__main__":
    main()
