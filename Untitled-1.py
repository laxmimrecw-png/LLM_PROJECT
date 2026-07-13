
import torch
from transformers import ( AutoTokenizer,AutoModelForCausalLM,
    Trainer,TrainingArguments,DataCollatorForLanguageModeling
)


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


model = AutoModelForCausalLM.from_pretrained(
    model_name
)
import pandas as pd

data = [
    {"Question": "What is real estate?", "Answer": "Real estate refers to land and any buildings or structures on it."},
    {"Question": "What is a property?", "Answer": "A property is a piece of land or a building owned by someone."},
    {"Question": "What is residential property?", "Answer": "Residential property is used for living, such as houses and apartments."},
    {"Question": "What is commercial property?", "Answer": "Commercial property is used for business purposes like offices and shops."},
    {"Question": "What is industrial property?", "Answer": "Industrial property includes factories, warehouses, and manufacturing units."},
    {"Question": "What is agricultural land?", "Answer": "Agricultural land is used for farming and cultivation."},
    {"Question": "What is a plot?", "Answer": "A plot is a piece of land without construction."},
    {"Question": "What is an apartment?", "Answer": "An apartment is a residential unit in a multi-story building."},
    {"Question": "What is a villa?", "Answer": "A villa is an independent luxury house with private space."},
    {"Question": "What is a duplex house?", "Answer": "A duplex is a house with two connected floors."},
    {"Question": "What is a penthouse?", "Answer": "A penthouse is a luxury apartment on the top floor."},
    {"Question": "What is carpet area?", "Answer": "Carpet area is the usable floor area inside a property."},
    {"Question": "What is built-up area?", "Answer": "Built-up area includes carpet area plus walls and balconies."},
    {"Question": "What is super built-up area?", "Answer": "It includes built-up area and common areas like lifts and corridors."},
    {"Question": "What is market value?", "Answer": "Market value is the estimated price of a property in the current market."},
    {"Question": "What is resale property?", "Answer": "A resale property is previously owned and sold again."},
    {"Question": "What is a new property?", "Answer": "A new property is directly sold by the builder for the first time."},
    {"Question": "What is a lease?", "Answer": "A lease is an agreement allowing someone to use property for a fixed period."},
    {"Question": "What is rent?", "Answer": "Rent is the amount paid to use someone else's property."},
    {"Question": "What is a landlord?", "Answer": "A landlord owns the property being rented."},
    {"Question": "What is a tenant?", "Answer": "A tenant rents and occupies the property."},
    {"Question": "What is a mortgage?", "Answer": "A mortgage is a loan used to buy property."},
    {"Question": "What is EMI?", "Answer": "EMI is the fixed monthly payment for a loan."},
    {"Question": "What is down payment?", "Answer": "Down payment is the initial amount paid while purchasing property."},
    {"Question": "What is a home loan?", "Answer": "A home loan is borrowed money to purchase a house."},
    {"Question": "What is property tax?", "Answer": "Property tax is charged by local authorities on property owners."},
    {"Question": "What is stamp duty?", "Answer": "Stamp duty is a government tax paid during property registration."},
    {"Question": "What is property registration?", "Answer": "It is the legal process of recording ownership with the government."},
    {"Question": "What is a sale deed?", "Answer": "A sale deed is a legal document proving property ownership transfer."},
    {"Question": "What is title deed?", "Answer": "It proves legal ownership of a property."},
    {"Question": "What is an encumbrance certificate?", "Answer": "It shows whether the property has any legal or financial liabilities."},
    {"Question": "What is RERA?", "Answer": "RERA regulates the real estate sector and protects buyers."},
    {"Question": "What is possession?", "Answer": "Possession is when the buyer officially receives the property."},
    {"Question": "What is occupancy certificate?", "Answer": "It certifies that a building is ready and safe to occupy."},
    {"Question": "What is completion certificate?", "Answer": "It confirms that construction follows approved plans."},
    {"Question": "What is freehold property?", "Answer": "The owner has complete ownership rights over the property."},
    {"Question": "What is leasehold property?", "Answer": "The owner has rights only for the lease period."},
    {"Question": "What is appreciation?", "Answer": "Appreciation is the increase in property value over time."},
    {"Question": "What is depreciation?", "Answer": "Depreciation is the decrease in property value."},
    {"Question": "What is ROI in real estate?", "Answer": "ROI measures the profit earned from a property investment."},
    {"Question": "How do I buy a house?", "Answer": "Select a property, verify documents, arrange finance, register, and take possession."},
    {"Question": "How do I sell my property?", "Answer": "List the property, find a buyer, complete documentation, and register the sale."},
    {"Question": "How do I calculate EMI?", "Answer": "EMI depends on loan amount, interest rate, and loan tenure."},
    {"Question": "What documents are required to buy property?", "Answer": "Sale deed, ID proof, address proof, tax receipts, and approval documents."},
    {"Question": "How can I verify property ownership?", "Answer": "Check the title deed and government land records."},
    {"Question": "What is a brokerage fee?", "Answer": "It is the commission paid to a real estate agent."},
    {"Question": "What is a real estate agent?", "Answer": "A professional who helps buy, sell, or rent properties."},
    {"Question": "What is a builder?", "Answer": "A builder develops residential or commercial properties."},
    {"Question": "What is under-construction property?", "Answer": "A property that is still being built."},
    {"Question": "What is ready-to-move property?", "Answer": "A completed property available for immediate occupancy."},
    {"Question": "What are maintenance charges?", "Answer": "Fees paid for maintaining common facilities."},
    {"Question": "What is a gated community?", "Answer": "A residential area with controlled entry and shared amenities."},
    {"Question": "What is society maintenance fee?", "Answer": "Charges collected for community maintenance."},
    {"Question": "What is a floor plan?", "Answer": "A layout showing room arrangement and dimensions."},
    {"Question": "What is a site plan?", "Answer": "A drawing showing property boundaries and structures."},
    {"Question": "What is property valuation?", "Answer": "Estimating the current market value of a property."},
    {"Question": "What factors affect property prices?", "Answer": "Location, amenities, demand, infrastructure, and market trends."},
    {"Question": "Why is location important?", "Answer": "Better locations usually provide higher value and convenience."},
    {"Question": "What is rental yield?", "Answer": "Annual rental income divided by property value."},
    {"Question": "What is capital gain?", "Answer": "Profit earned from selling property at a higher price."},
    {"Question": "What is a co-owner?", "Answer": "A person who jointly owns property."},
    {"Question": "Can NRIs buy property in India?", "Answer": "Yes, NRIs can buy most residential and commercial properties in India."},
    {"Question": "What is a power of attorney?", "Answer": "A legal document allowing another person to act on your behalf."},
    {"Question": "What is mutation of property?", "Answer": "Updating ownership records after purchase."},
    {"Question": "What is legal verification?", "Answer": "Checking property documents for legal validity."},
    {"Question": "What is a survey number?", "Answer": "A unique government identification number for land."},
    {"Question": "What is a Khata certificate?", "Answer": "A document showing property details for taxation."},
    {"Question": "What is an allotment letter?", "Answer": "A builder's document confirming property allocation."},
    {"Question": "What is a booking amount?", "Answer": "Initial payment made to reserve a property."},
    {"Question": "What is a possession letter?", "Answer": "A document confirming handover of property."},
    {"Question": "What is an agreement to sell?", "Answer": "A legal agreement before the final sale deed."},
    {"Question": "What is foreclosure?", "Answer": "Closing a loan before its scheduled end date."},
    {"Question": "What is prepayment?", "Answer": "Paying part of a loan before the due date."},
    {"Question": "What is a floating interest rate?", "Answer": "An interest rate that changes with market conditions."},
    {"Question": "What is a fixed interest rate?", "Answer": "An interest rate that remains constant during the loan period."},
    {"Question": "What is a credit score?", "Answer": "A score representing a person's creditworthiness."},
    {"Question": "Why is a credit score important?", "Answer": "It affects loan approval and interest rates."},
    {"Question": "What is a loan tenure?", "Answer": "The duration to repay a loan."},
    {"Question": "What is property insurance?", "Answer": "Insurance protecting property from damage or loss."},
    {"Question": "What is home insurance?", "Answer": "Insurance covering a house and its contents."},
    {"Question": "What is Vastu?", "Answer": "Traditional architectural principles followed in many Indian homes."},
    {"Question": "What is a smart home?", "Answer": "A home with automated technology and connected devices."},
    {"Question": "What amenities increase property value?", "Answer": "Parking, security, parks, gyms, and swimming pools."},
    {"Question": "How do I find nearby schools?", "Answer": "Use online maps or local property portals."},
    {"Question": "How do I find nearby hospitals?", "Answer": "Search on maps or local directories."},
    {"Question": "What is covered parking?", "Answer": "Parking space protected by a roof or structure."},
    {"Question": "What is open parking?", "Answer": "Parking space without a roof."},
    {"Question": "Can I negotiate property price?", "Answer": "Yes, buyers often negotiate based on market conditions."},
    {"Question": "How long does registration take?", "Answer": "Usually completed within a few days depending on local authorities."},
    {"Question": "What is token advance?", "Answer": "A small payment showing purchase intention."},
    {"Question": "Can I cancel a booking?", "Answer": "It depends on the builder's cancellation policy."},
    {"Question": "What is a possession date?", "Answer": "The date when the buyer receives the property."},
    {"Question": "What is an investment property?", "Answer": "Property purchased mainly to earn income or appreciation."},
    {"Question": "Which property is best for investment?", "Answer": "It depends on budget, location, and investment goals."},
    {"Question": "Is buying better than renting?", "Answer": "It depends on financial situation and long-term plans."},
    {"Question": "How can I estimate property value?", "Answer": "Compare similar properties and consult valuation experts."},
    {"Question": "What is a real estate portfolio?", "Answer": "A collection of multiple real estate investments."},
    {"Question": "How do I contact a property owner?", "Answer": "Through listings, agents, or provided contact details."},
    {"Question": "What should I inspect before buying?", "Answer": "Check structure, documents, utilities, and neighborhood."},
    {"Question": "How do I avoid property fraud?", "Answer": "Verify documents and use legal experts before purchasing."},
    {"Question": "What is the first step in buying property?", "Answer": "Define your budget and search for suitable properties."},
    {"Question": "How do I compare two properties?", "Answer": "Compare price, location, size, amenities, and future value."},
    {"Question": "Can I get a home loan for resale property?", "Answer": "Yes, if the property meets lender requirements."},
    {"Question": "What is the best time to invest in property?", "Answer": "When prices, demand, and your financial situation are favorable."}
]

df = pd.DataFrame(data)
print(df)

from datasets import load_dataset
df.to_json("train_data.json",orient="records",indent=4)
dataset=load_dataset("json",data_files="train_data.json")
print(dataset)

def formatting(data):
       return {
               "text":f"""### Question:{data["Question"]}
               ### Answer:{data["Answer"]}"""
  }
dataset=dataset["train"].map(formatting)
print(dataset)

tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
def tokenize(prompt):
  result=tokenizer(prompt["text"],padding="max_length",truncation=True,max_length=128)
  result["labels"] = result["input_ids"].copy()
  return result
dataset=dataset.map(tokenize)
print(dataset)


from peft import (LoraConfig,get_peft_model)
config=LoraConfig(r=8,lora_alpha=16,
                    target_modules=["q_proj","v_proj"],
                  lora_dropout=0.1,bias="none",task_type="CAUSAL_LM")
model=get_peft_model(model,config)
model.print_trainable_parameters()

training_args=TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True
)

trainer=Trainer (
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)
trainer.train()

model.save_pretrained("tinyllama_adapter")
tokenizer.save_pretrained("tinyllama_adapter")

device=next(model.parameters()).device
prompt="""### Question:
How do I find nearby schools
### Answer:
"""
inputs=tokenizer(prompt,return_tensors="pt")
inputs={k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
  output=model.generate(**inputs,max_new_tokens=100)
print(tokenizer.decode(output[0],skip_special_tokens=True)) 