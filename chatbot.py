def simple_chatbot(user_input):
    # Convert user input to lowercase for case-insensitive matching
    user_input = user_input.lower()

    # Define predefined rules and responses
    rules_responses = {
        "hello": "Hi there! How can I help you?",
        "how are you": "I'm just a computer program, but thanks for asking!",
        "goodbye": "Goodbye! Have a great day!",
        "name": "I'm just a simple chatbot. You can call me Luffy.",
        "weather": "I'm sorry, I don't have the capability to check the weather right now.",
        "default": "I'm not sure how to respond to that. Ask me something else!",
        "how was your day?": "Usual ig",
        "can you crack a joke?" : "Engineering amiright?"
    }

    # Check user input against predefined rules
    for rule, response in rules_responses.items():
        if rule in user_input:
            return response

    # If no match is found, use the default response
    return rules_responses["default"]

# Main loop for the chatbot
while True:
    user_input = input("User: ")
    
    # Check for an exit command
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Get chatbot response based on user input
    chatbot_response = simple_chatbot(user_input)
    print("Chatbot:", chatbot_response)
