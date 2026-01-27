# Bandwidth + OpenAI Realtime Websockets Integration - Python # Table of Contents

* [Description](#description)
* [Pre-Requisites](#pre-requisites)
* [Environmental Variables](#environmental-variables)
* [Running the Application](#running-the-application)
* [Callback URLs](#callback-urls)
  * [Ngrok](#ngrok)

# Description

This is a sample application that demonstrates how to use the Bandwidth's Programmable Voice API with OpenAI's Realtime Websocket Interface to create a real-time AI-powered voice assistant.

# Pre-Requisites

In order to use the Bandwidth API users need to set up the appropriate application at the [Bandwidth App](https://dashboard.bandwidth.com/) and create API tokens.

To create an application log into the [Bandwidth App](https://dashboard.bandwidth.com/) and navigate to the `Applications` tab.  Fill out the **New Application** form selecting `Voice`.  All Bandwidth services require publicly accessible Callback URLs, for more information on how to set one up see [Callback URLs](#callback-urls).

For more information about API credentials see our [Account Credentials](https://dev.bandwidth.com/docs/account/credentials) page.

# Environmental Variables

The sample app uses the below environmental variables. Create a `.env` file in the root directory of the project with the following variables:

```sh
BW_ACCOUNT_ID=your_bandwidth_account_id
BW_USERNAME=your_bandwidth_username
BW_PASSWORD=your_bandwidth_password
OPENAI_API_KEY=your_openai_api_key
TRANSFER_TO=+19195551212  # The phone number to transfer the call to (in E.164 format)
BASE_URL=https://your-ngrok-url.ngrok.io  # The base URL from ngrok (see Callback URLs section)
LOG_LEVEL=INFO  # (optional) The logging level for the application (default: INFO)
LOCAL_PORT=3000  # (optional) The local port for the application (default: 5000)
```

# Running the Application

This application is built using Python 3.13. Follow these steps to run the application locally:

1. **Create a virtual environment:**
```sh
python -m venv .venv
```

2. **Activate the virtual environment:**
```sh
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

3. **Install the required packages:**
```sh
cd app
pip install -r requirements.txt
```

4. **Set up your environment variables:**
   - Create a `.env` file in the root directory with the required variables (see [Environmental Variables](#environmental-variables) section)

5. **Set up ngrok for callback URLs:**
   - Follow the instructions in the [Ngrok](#ngrok) section below to expose your local server

6. **Run the application:**
```sh
python main.py
```

# Callback URLs

For a detailed introduction, check out our [Bandwidth Voice Webhooks](https://dev.bandwidth.com/docs/voice/programmable-voice/webhooks) page.

Below are the callback paths exposed by this application:
* `/health`
* `/webhooks/bandwidth/voice/initiate`
* `/webhooks/bandwidth/voice/status`

## Ngrok

A simple way to set up a local callback URL for testing is to use the free tool [ngrok](https://ngrok.com/).  
After you have downloaded and installed `ngrok` run the following command to open a public tunnel to your port (`$LOCAL_PORT`)

```sh
ngrok http $LOCAL_PORT
```

You can view your public URL at `http://127.0.0.1:4040` after ngrok is running.  You can also view the status of the tunnel and requests/responses here.
