# render.yaml

services:
  - type: web
    name: hebrew-semantle-api # A slightly new name to help avoid caching issues
    runtime: python
    plan: free
    region: oregon
    
    # The build command is now as simple as possible.
    buildCommand: "pip install -r requirements.txt"
    
    # The start command is unchanged.
    startCommand: "uvicorn app:app --host 0.0.0.0 --port 10000"
    
    healthCheckPath: /
    envVars:
      - key: PYTHON_VERSION
        value: '3.11'
      - key: PYTHONUNBUFFERED
        value: '1'
      - key: ADMIN_PASSWORD
        sync: false
      - key: GOOGLE_DRIVE_FOLDER_ID
        sync: false
      - key: GOOGLE_SERVICE_ACCOUNT_JSON
        sync: false
