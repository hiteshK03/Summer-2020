import os
class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you-will-never-guess"
    UPLOAD_FOLDER = "image-file"
    CSRF_ENABLED = True
    DEBUG = False
    
    # Enter your API Key and Custom Classifier ID below
    API_KEY = "SYV4GEFYaOW0Li9vpCtLz9PLFDkvkn8A5rt57L86ePnU"
    CLASSIFIER_ID = "DefaultCustomModel_481980631"
