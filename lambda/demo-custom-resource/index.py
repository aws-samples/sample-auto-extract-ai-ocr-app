import boto3
import json
import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

DSQL_ENDPOINT = os.environ.get("DSQL_ENDPOINT", "")
DSQL_REGION = os.environ.get("DSQL_REGION", "")


def handler(event, context):
    print(f"Event: {json.dumps(event)}")

    if event['RequestType'] == 'Delete':
        return {'PhysicalResourceId': 'demo-data'}

    if event['RequestType'] != 'Create':
        return {'PhysicalResourceId': 'demo-data'}

    try:
        region = os.environ.get('AWS_REGION')
        dynamodb = boto3.resource('dynamodb', region_name=region)

        # Get table names from properties
        customers_table_name = event['ResourceProperties']['CustomersTableName']
        schemas_table_name = event['ResourceProperties']['SchemasTableName']

        # Insert demo customers
        insert_demo_customers(dynamodb, customers_table_name)

        # Insert demo usecase (DynamoDB)
        insert_demo_usecase(dynamodb, schemas_table_name)

        # Insert demo usecase + tool bindings (DSQL)
        if DSQL_ENDPOINT:
            insert_demo_usecase_dsql()

        return {'PhysicalResourceId': 'demo-data'}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


def insert_demo_customers(dynamodb, table_name):
    """Insert demo customer data"""
    table = dynamodb.Table(table_name)

    customers = [
        {
            'customer_id': 'CUST001',
            'customer_name': 'サンプル株式会社',
            'postal_code': '〒123-4567',
            'address': '東京都目黒区上目黒1-2-3 サンプルビル 6階',
            'phone': '03-1234-5679',
            'email': 'info@sample.co.jp',
            'contact_person': 'サンプル太郎'
        },
        {
            'customer_id': 'CUST002',
            'customer_name': 'テスト商事株式会社',
            'postal_code': '〒100-0001',
            'address': '東京都千代田区千代田1-1-1',
            'phone': '03-0000-0001',
            'email': 'contact@test-corp.co.jp',
            'contact_person': '田中花子'
        },
        {
            'customer_id': 'CUST003',
            'customer_name': '株式会社デモカンパニー',
            'postal_code': '〒150-0001',
            'address': '東京都渋谷区神宮前1-1-1',
            'phone': '03-9999-9999',
            'email': 'info@demo-company.jp',
            'contact_person': '山田次郎'
        }
    ]

    for customer in customers:
        table.put_item(Item=customer)
        print(f"Inserted customer: {customer['customer_id']}")


def insert_demo_usecase(dynamodb, table_name):
    """Insert demo invoice usecase"""
    table = dynamodb.Table(table_name)

    # Load demo schema from file
    with open('demo_invoice_schema.json', 'r', encoding='utf-8') as f:
        demo_schema = json.load(f)

    # Create demo usecase
    demo_usecase = {
        'schema_type': 'app',
        'name': 'demo_invoice',
        'display_name': '(demo)請求書',
        'description': 'デモ用請求書抽出ユースケース',
        'fields': demo_schema,
        'input_methods': {
            'file_upload': True,
            's3_sync': False
        },
        'agent_enabled': True,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat()
    }

    table.put_item(Item=demo_usecase)
    print(f"Inserted demo usecase: {demo_usecase['name']}")


def _get_dsql_connection():
    """Get DSQL connection with IAM auth"""
    client = boto3.client("dsql", region_name=DSQL_REGION)
    token = client.generate_db_connect_admin_auth_token(DSQL_ENDPOINT, DSQL_REGION)
    return psycopg2.connect(
        host=DSQL_ENDPOINT,
        port=5432,
        user="admin",
        password=token,
        dbname="postgres",
        sslmode="require",
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def insert_demo_usecase_dsql():
    """Insert demo usecase and bind all available tools in DSQL"""
    conn = _get_dsql_connection()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Check if demo usecase already exists
            cur.execute("SELECT id FROM usecases WHERE app_name = %s", ("demo_invoice",))
            existing = cur.fetchone()
            if existing:
                usecase_id = str(existing["id"])
                print(f"Demo usecase already exists: {usecase_id}")
            else:
                # Insert usecase (created_by uses a system placeholder UUID)
                usecase_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO usecases (id, app_name, created_by) VALUES (%s, %s, %s)",
                    (usecase_id, "demo_invoice", "00000000-0000-0000-0000-000000000000"),
                )
                print(f"Inserted demo usecase in DSQL: {usecase_id}")

            # Bind all active tools to this usecase
            cur.execute("SELECT id, name FROM tools WHERE is_active = true")
            tools = cur.fetchall()

            for tool in tools:
                tool_id = str(tool["id"])
                cur.execute(
                    """INSERT INTO usecase_tools (usecase_id, tool_id)
                    VALUES (%s, %s)
                    ON CONFLICT (usecase_id, tool_id) DO NOTHING""",
                    (usecase_id, tool_id),
                )

            print(f"Bound {len(tools)} tools to demo usecase")

            # Grant viewer permission to the 'all' group so every signed-up
            # user can access the demo usecase. The 'all' group is created by
            # dsql-admin seed; this resource is wired to depend on it so the
            # group should always exist by the time this runs. If it does not,
            # we log a warning and skip rather than failing the deployment.
            cur.execute("SELECT id FROM groups WHERE name = 'all'")
            all_group = cur.fetchone()
            if all_group:
                cur.execute(
                    """INSERT INTO group_usecases (group_id, usecase_id, permission)
                    VALUES (%s, %s, 'viewer')
                    ON CONFLICT (group_id, usecase_id) DO NOTHING""",
                    (str(all_group["id"]), usecase_id),
                )
                print(f"Granted 'all' group viewer permission on demo usecase")
            else:
                print("WARNING: 'all' group not found; skipping group_usecases grant")

    finally:
        conn.close()
