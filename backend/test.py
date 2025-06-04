from appbuilder.core.console.appbuilder_client import AppBuilderClient

APP_ID = "cfa54f11-aa5f-42ac-84e5-971f7c4d9a8b"
APPBUILDER_TOKEN = "bce-v3/ALTAK-m3ujKSSAYbObjasQymF6i/8b88495e444a19a6be22ea0b8199b557f5f5d716"

# 创建AppBuilder客户端，传递secret_key而非api_key
client = AppBuilderClient(APP_ID, secret_key=APPBUILDER_TOKEN)

print(client)
