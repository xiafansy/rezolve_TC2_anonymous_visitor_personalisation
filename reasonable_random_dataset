import random
import pandas as pd

# ==========================
# Configuration
# ==========================

NUM_SESSIONS = 1000

intents = [
    "Goal-driven",
    "Explorer",
    "Research",
    "Price-sensitive"
]

times = ["Morning", "Afternoon", "Evening"]

devices = ["Mobile", "Desktop", "Tablet"]

categories = [
    "Running",
    "Fashion",
    "Electronics",
    "Beauty",
    "Home",
    "Sale"
]

search_queries = {
    "Running": [
        "running shoes",
        "nike shoes",
        "adidas trainers"
    ],
    "Fashion": [
        "summer dress",
        "black jacket",
        "white trainers"
    ],
    "Electronics": [
        "wireless headphones",
        "gaming mouse",
        "monitor"
    ],
    "Beauty": [
        "lipstick",
        "skincare",
        "perfume"
    ],
    "Home": [
        "desk lamp",
        "chair",
        "coffee table"
    ],
    "Sale": [
        "discount",
        "clearance",
        "cheap deals"
    ]
}

dataset = []

# ==========================
# Generate Sessions
# ==========================

for session in range(1, NUM_SESSIONS + 1):

    intent = random.choice(intents)

    # -------------------------
    # Goal-driven
    # -------------------------
    if intent == "Goal-driven":

        referrer = random.choices(
            ["Google", "Direct"],
            weights=[0.9, 0.1]
        )[0]

        device = random.choice(["Mobile", "Desktop"])

        category = random.choice(
            ["Running", "Electronics", "Beauty"]
        )

        search_used = True

        search_query = random.choice(
            search_queries[category]
        )

        scroll_depth = random.randint(20, 50)

        product_views = random.randint(2, 6)

        filter_used = random.random() < 0.4

        sort_type = random.choice([
            "Relevance",
            "Newest"
        ])

        duration = random.randint(60, 300)

        add_to_cart = random.random() < 0.75

        purchase = add_to_cart and random.random() < 0.60

    # -------------------------
    # Explorer
    # -------------------------
    elif intent == "Explorer":

        referrer = random.choices(
            ["Instagram", "Facebook", "Direct"],
            weights=[0.6, 0.2, 0.2]
        )[0]

        device = random.choice(devices)

        category = random.choice(categories)

        search_used = False

        search_query = ""

        scroll_depth = random.randint(80, 100)

        product_views = random.randint(10, 20)

        filter_used = False

        sort_type = "Trending"

        duration = random.randint(300, 900)

        add_to_cart = random.random() < 0.15

        purchase = False

    # -------------------------
    # Research
    # -------------------------
    elif intent == "Research":

        referrer = random.choice(
            ["Google", "Direct", "Email"]
        )

        device = "Desktop"

        category = random.choice(categories)

        search_used = random.random() < 0.7

        if search_used:
            search_query = random.choice(
                search_queries[category]
            )
        else:
            search_query = ""

        scroll_depth = random.randint(50, 80)

        product_views = random.randint(8, 15)

        filter_used = True

        sort_type = random.choice([
            "Highest Rated",
            "Price Low-High"
        ])

        duration = random.randint(400, 1000)

        add_to_cart = random.random() < 0.35

        purchase = add_to_cart and random.random() < 0.25

    # -------------------------
    # Price-sensitive
    # -------------------------
    else:

        referrer = random.choice(
            ["Email", "Google", "Direct"]
        )

        device = random.choice(devices)

        category = "Sale"

        search_used = random.random() < 0.4

        if search_used:
            search_query = "discount"
        else:
            search_query = ""

        scroll_depth = random.randint(40, 70)

        product_views = random.randint(5, 10)

        filter_used = True

        sort_type = "Price Low-High"

        duration = random.randint(200, 600)

        add_to_cart = random.random() < 0.60

        purchase = add_to_cart and random.random() < 0.45

    # -------------------------
    # Save Session
    # -------------------------

    dataset.append({

        "Session_ID": session,

        "Intent": intent,

        "Referrer": referrer,

        "Device": device,

        "Time_of_Day": random.choice(times),

        "Category": category,

        "Search_Used": search_used,

        "Search_Query": search_query,

        "Scroll_Depth": scroll_depth,

        "Product_Views": product_views,

        "Filter_Used": filter_used,

        "Sort_Type": sort_type,

        "Session_Duration_sec": duration,

        "Add_to_Cart": add_to_cart,

        "Purchase": purchase

    })

# ==========================
# Export
# ==========================

df = pd.DataFrame(dataset)

print(df.head())

print("\nDataset Shape:", df.shape)

df.to_csv("anonymous_sessions.csv", index=False)

print("\nCSV file saved as anonymous_sessions.csv")
