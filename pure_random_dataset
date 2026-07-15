import random
import pandas as pd


referrers = [
    "Google",
    "Instagram",
    "Direct",
    "Email",
    "Facebook"
]

devices = [
    "Mobile",
    "Desktop",
    "Tablet"
]

categories = [
    "Running",
    "Fashion",
    "Electronics",
    "Sale",
    "Beauty",
    "Home"
]

intents = [
    "Goal-driven",
    "Explorer",
    "Research",
    "Price-sensitive"
]


  
user = {
    "Session_ID": 1,
    "Referrer": random.choice(referrers),
    "Device": random.choice(devices),
    "Category": random.choice(categories),
    "Scroll_Depth": random.randint(10,100),
    "Product_Views": random.randint(1,15)
}



data = []

for i in range(1000):

    user = {
        "Session_ID": i+1,
        "Referrer": random.choice(referrers),
        "Device": random.choice(devices),
        "Category": random.choice(categories),
        "Scroll_Depth": random.randint(10,100),
        "Product_Views": random.randint(1,15)
    }

    data.append(user)


      
df = pd.DataFrame(data)

df.head()


      
df.to_csv("anonymous_users.csv",index=False)
