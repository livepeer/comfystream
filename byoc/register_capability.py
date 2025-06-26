import requests, time

#note: price is 0 for the capability registered so will be able to use with wallets with no eth deposit/reserve
#      if price is set above 0 will need to use an on chain Orchestrator and a Gateway with a Deposit/Reserve
data = {
   "name": "comfystream-video",
   "url": "http://byoc_reverse_text:8889",
   "capacity": 1,
   "price_per_unit": 0,
   "price_scaling": 1,
   "currency": "wei"
}

headers = {
    "Authorization": "orch-secret"
}

for i in range(10):
    #wait 1 second then try
    time.sleep(1)

    try:
        registered = requests.post("https://byoc_orchestrator:8935/capability/register", json=data, headers=headers, verify=False)
        if registered.status_code == 200:
            break
        else:
            print(f"registration not completed: {registered.text}")
    except Exception as e:
        pass   