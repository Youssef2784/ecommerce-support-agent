"""Generate synthetic order data using Faker.

Produces >=200 orders with realistic e-commerce fields.
Deterministic (seeded) so the dataset is reproducible.

Run: python data/orders/generate_orders.py
"""

import json
import random
from datetime import timedelta
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# --- Configuration ---
NUM_ORDERS = 220
NUM_CUSTOMERS = 50

STATUSES = ["delivered", "in-transit", "delayed", "returned", "cancelled"]
STATUS_WEIGHTS = [0.40, 0.25, 0.15, 0.10, 0.10]  # realistic distribution

PRODUCTS = {
    "Electronics": [
        ("Wireless Earbuds", 29.99),
        ("USB-C Hub", 34.99),
        ("Portable Charger", 24.99),
        ("Smart Watch", 149.99),
        ("Bluetooth Speaker", 49.99),
        ("Webcam HD", 59.99),
        ("Mechanical Keyboard", 89.99),
        ("Phone Case", 14.99),
    ],
    "Home & Kitchen": [
        ("Air Fryer", 79.99),
        ("Blender", 39.99),
        ("Coffee Maker", 54.99),
        ("Cutting Board Set", 19.99),
        ("Knife Set", 44.99),
        ("Water Bottle", 12.99),
    ],
    "Clothing": [
        ("Running Shoes", 69.99),
        ("Cotton T-Shirt", 19.99),
        ("Hoodie", 39.99),
        ("Denim Jacket", 59.99),
        ("Baseball Cap", 14.99),
    ],
    "Books": [
        ("Python Crash Course", 29.99),
        ("Clean Code", 34.99),
        ("Designing Data-Intensive Apps", 39.99),
        ("The Pragmatic Programmer", 42.99),
    ],
}

CATEGORIES = list(PRODUCTS.keys())


def generate_customer_ids(n: int) -> list[str]:
    """Create stable customer IDs."""
    return [f"CUST-{1000 + i}" for i in range(n)]


def generate_orders(customer_ids: list[str], n: int) -> list[dict]:
    """Generate n synthetic orders across the given customers."""
    orders = []
    for i in range(n):
        customer_id = random.choice(customer_ids)
        category = random.choice(CATEGORIES)
        # Pick 1-3 items from that category
        num_items = random.randint(1, 3)
        items = random.choices(PRODUCTS[category], k=num_items)

        order_date = fake.date_time_between(start_date="-90d", end_date="now")
        status = random.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]

        # Estimated delivery: 3-10 days after order
        est_delivery = order_date + timedelta(days=random.randint(3, 10))

        order = {
            "order_id": f"ORD-{10000 + i}",
            "customer_id": customer_id,
            "status": status,
            "items": [{"name": name, "price": price} for name, price in items],
            "category": category,
            "total": round(sum(p for _, p in items), 2),
            "order_date": order_date.isoformat(),
            "estimated_delivery": est_delivery.strftime("%Y-%m-%d"),
            "shipping_address": fake.address().replace("\n", ", "),
        }

        # Add tracking number for shipped orders
        if status in ("delivered", "in-transit", "delayed"):
            order["tracking_number"] = fake.bothify("TRK-####-????").upper()

        # Add return reason for returned orders
        if status == "returned":
            order["return_reason"] = random.choice([
                "Wrong size", "Defective product", "Changed my mind",
                "Not as described", "Arrived too late",
            ])

        # Add cancellation reason for cancelled orders
        if status == "cancelled":
            order["cancellation_reason"] = random.choice([
                "Customer requested", "Payment failed",
                "Out of stock", "Duplicate order",
            ])

        orders.append(order)

    return orders


if __name__ == "__main__":
    customer_ids = generate_customer_ids(NUM_CUSTOMERS)
    orders = generate_orders(customer_ids, NUM_ORDERS)

    out_path = Path(__file__).parent / "orders.json"
    with open(out_path, "w") as f:
        json.dump(orders, f, indent=2)

    # Print summary
    from collections import Counter
    status_counts = Counter(o["status"] for o in orders)
    print(f"Generated {len(orders)} orders for {NUM_CUSTOMERS} customers")
    print(f"Status distribution: {dict(status_counts)}")
    print(f"Saved to {out_path}")
