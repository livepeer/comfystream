import requests, time
import argparse

#note: price is 0 for the capability registered so will be able to use with wallets with no eth deposit/reserve
#      if price is set above 0 will need to use an on chain Orchestrator and a Gateway with a Deposit/Reserve

# Parse command line arguments
parser = argparse.ArgumentParser(description='Register capabilities with orchestrator')
parser.add_argument('--orch-host', 
                    default='byoc_orchestrator', 
                    help='Orchestrator host (default: byoc_orchestrator)')
parser.add_argument('--orch-port', 
                    default='8935', 
                    help='Orchestrator port (default: 8935)')
args = parser.parse_args()

orch_host = args.orch_host
orch_port = args.orch_port

# Define multiple capabilities to register
capabilities = [
    {
        "name": "comfystream-video",
        "url": f"http://{orch_host}:8889",
        "capacity": 1,
        "price_per_unit": 0,
        "price_scaling": 1,
        "currency": "wei"
    },
    {
        "name": "whip-ingest",
        "url": f"http://{orch_host}:8889",
        "capacity": 1,
        "price_per_unit": 0,
        "price_scaling": 1,
        "currency": "wei"
    },
    {
        "name": "whep-subscribe",
        "url": f"http://{orch_host}:8889",
        "capacity": 1,
        "price_per_unit": 0,
        "price_scaling": 1,
        "currency": "wei"
    }
]

headers = {
    "Authorization": "orch-secret"
}

# Register each capability
for capability in capabilities:
    print(f"Registering capability: {capability['name']}")
    
    for i in range(10):
        #wait 1 second then try
        time.sleep(1)

        try:
            registered = requests.post(f"https://{orch_host}:{orch_port}/capability/register", json=capability, headers=headers, verify=False)
            if registered.status_code == 200:
                print(f"Successfully registered capability: {capability['name']}")
                break
            else:
                print(f"registration not completed for {capability['name']}: {registered.text}")
        except Exception as e:
            print(f"Error registering {capability['name']}: {e}")
            pass   