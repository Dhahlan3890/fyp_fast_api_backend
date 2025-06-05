from gradio_client import Client, handle_file

client = Client("Dhahlan2000/predict_freshness_and_ripeness")
result = client.predict(
		image=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		api_name="/predict"
)
print(result)