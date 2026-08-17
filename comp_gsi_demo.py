import boto3
client = boto3.client(
    'dynamodb',
    endpoint_url='http://127.0.0.1:8000',
    region_name='us-east-1',
    aws_access_key_id='dummy',
    aws_secret_access_key='dummy',
)

_ = client.create_table(
    TableName='DealershipCars',
    KeySchema=[
        {'AttributeName': 'car_id', 'KeyType': 'HASH'},
        {'AttributeName': 'price', 'KeyType': 'RANGE'},
    ],
    AttributeDefinitions=[
        {'AttributeName': 'car_id', 'AttributeType': 'S'},
        {'AttributeName': 'price', 'AttributeType': 'N'},
        {'AttributeName': 'color', 'AttributeType': 'S'},
        {'AttributeName': 'horsepower', 'AttributeType': 'N'},
    ],
    # Frequently when married couples shop for a car we need to satisfy 2
    # equally deal breaking preferences: color and horsepower.
    # Additionally we need to bound results by price.
    GlobalSecondaryIndexes=[
        {
            'IndexName': 'MarriedCoupleGSI',
            'KeySchema': [
                {'AttributeName': 'color', 'KeyType': 'HASH'},
                {'AttributeName': 'horsepower', 'KeyType': 'HASH'},
                {'AttributeName': 'price', 'KeyType': 'RANGE'},
            ],
            'Projection': {'ProjectionType': 'ALL'},
        },
    ],
    BillingMode='PAY_PER_REQUEST',
)

_ = client.put_item(
    TableName='DealershipCars',
    Item={
        'car_id': {'S': '01'},
        'price': {'N': '20000'},
        'name': {'S': 'Toyota'},
        'color': {'S': 'red'},
        'horsepower': {'N': '100'},
    }
)

_ = client.put_item(
    TableName='DealershipCars',
    Item={
        'car_id': {'S': '02'},
        'price': {'N': '15000'},
        'name': {'S': 'Honda'},
        # We don't have a color for this Honda yet.
        'horsepower': {'N': '80'},
    }
)

_ = client.put_item(
    TableName='DealershipCars',
    Item={
        'car_id': {'S': '03'},
        'price': {'N': '30000'},
        'name': {'S': 'BMW'},
        'color': {'S': 'blue'},
        'horsepower': {'N': '100'},
    }
)

_ = client.put_item(
    TableName='DealershipCars',
    Item={
        'car_id': {'S': '04'},
        'price': {'N': '40000'},
        'name': {'S': 'Mercedes'},
        'color': {'S': 'red'},
        'horsepower': {'N': '120'},
    }
)

response = client.query(
    TableName='DealershipCars',
    IndexName='MarriedCoupleGSI',
    KeyConditionExpression='color = :color AND horsepower = :hp AND price < :price',
    ExpressionAttributeValues={
        ':color': {'S': 'red'},
        ':hp': {'N': '100'},
        ':price': {'N': '25000'},
    },
)

import json
print(json.dumps(response['Items'], indent=2))
