"""
Locust load generator for Online Boutique.
Simulates realistic user behavior: browse, view product, add to cart, checkout.
"""

import random
from locust import HttpUser, task, between

# Product IDs from Online Boutique seed data
PRODUCT_IDS = [
    "OLJCESPC7Z",
    "66VCHSJNUP",
    "1YMWWN1N4O",
    "L9ECAV7KIM",
    "2ZYFJ3GM2N",
    "0PUK6V6EV0",
    "LS4PSXUNUM",
    "HQTGWGPNH4",
    "9SIQT8TO72",
    "6E92ZM6F9J",
]

CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD"]


class BoutiqueUser(HttpUser):
    """Simulates a single Online Boutique user."""

    # Wait 1-5 seconds between requests (realistic think time)
    wait_time = between(1, 5)

    def on_start(self):
        """Initialize session with a currency preference."""
        currency = random.choice(CURRENCIES)
        self.client.get(f"/setCurrency?currency_code={currency}", name="/setCurrency")

    @task(5)
    def browse_home(self):
        """Load the home page (most common action)."""
        self.client.get("/", name="/ [Home]")

    @task(4)
    def view_product(self):
        """View a random product detail page."""
        pid = random.choice(PRODUCT_IDS)
        self.client.get(f"/product/{pid}", name="/product/[id]")

    @task(2)
    def add_to_cart(self):
        """View product then add it to cart."""
        pid = random.choice(PRODUCT_IDS)
        self.client.get(f"/product/{pid}", name="/product/[id]")

        self.client.post(
            "/cart",
            data={"product_id": pid, "quantity": "1"},
            name="/cart [POST add]",
        )

    @task(2)
    def view_cart(self):
        """View the shopping cart."""
        self.client.get("/cart", name="/cart [GET]")

    @task(1)
    def full_checkout(self):
        """Complete browse → add to cart → checkout flow."""
        pid = random.choice(PRODUCT_IDS)

        # Browse
        self.client.get("/", name="/ [Home]")
        self.client.get(f"/product/{pid}", name="/product/[id]")

        # Add to cart
        self.client.post(
            "/cart",
            data={"product_id": pid, "quantity": str(random.randint(1, 3))},
            name="/cart [POST add]",
        )

        # View cart
        self.client.get("/cart", name="/cart [GET]")

        # Checkout
        self.client.post(
            "/cart/checkout",
            data={
                "email": f"user{random.randint(1, 999)}@example.com",
                "street_address": f"{random.randint(1, 9999)} Main St",
                "zip_code": f"{random.randint(10000, 99999)}",
                "city": "Mountain View",
                "state": "CA",
                "country": "US",
                "credit_card_number": "4432-8015-6152-0454",
                "credit_card_expiration_month": "1",
                "credit_card_expiration_year": "2030",
                "credit_card_cvv": "623",
            },
            name="/cart/checkout [POST]",
        )

    @task(1)
    def switch_currency(self):
        """Switch display currency (triggers currencyservice calls)."""
        currency = random.choice(CURRENCIES)
        self.client.get(
            f"/setCurrency?currency_code={currency}", name="/setCurrency"
        )
