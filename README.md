# agentic assistant

this is a work in progress.

the vision for this is to become an agentic assistant that can take actions on your behalf without your explicit instructions. right now, it's basically a more capable siri with respect to calendar management and music playback. it's speech to text, text to speech.

uses gpt + elevenlabs + whisper.

by using two seperate llms, one to take action and one to respond to the user, the assistant is able to handle more nuanced and complex instructions than exisiting virtual assistants. indeed, it can even handle multiple completely seperate complex instructions at once.

currently, the calendar integration is with a custom made calendar class i built. it shouldn't bee too bad to swap it to use the google calendar api in the future though.

biggest challenges: latency, due to stt and tts nature of the bot. improving communication between the action llm and conversation llm. adding more functionality. making it multi-platform.

DISCLAIMER: this is kinda just a rough mvp of what i'm planning. that's why the code is so rough - just did it all in one go. i'm planning on cleaning this up and building a more capable version in the future, with all the improvements listed above.
