# ZoraAI

Uses Huggingface's 7b Parameter Satbility LLM, together with AutoGPT and BabyAGI. 

The model will continuously use different tools to solve the overall objective, as AutoGPT and BabyAGI both do. It uses different instances of the LLM with different roles to prioritize tasks.

A tool loader is included in the BabyAGI version (AutoGPT version is older). Through Langchain the LLM will call this tool loader if it finds itself unable to perform a certain action (i.e. Wants to send an email but doesn't have that API). Through using "plugnplai", the loader will search for an API that will allow it to perform the required task and add that API to its array of available tools. Thus, it allows it to grow and adapt to solve probelems it ordinarially would not have the data or the ability to perform

Currently the huggingface model is used, however this can easily be substituted for the OpenAI GPT4 api using Langchain which may increase accuracy and will decrease compusing resources dramatically

TODO:
 - Actually debug it properly
 - Add a separate LLM to allow the user to ask questions and give commands even while it's currently working on an objective
 - After this, voice control will be more natural and can be added using TTS and speach recognition
 - Finally, another LLM can be used to continuously monitor ambient conversation and trained to identify if it should be spoken to or not -> Thus avoiding the need for keyword activation
