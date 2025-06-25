#!/home/jetson/ros_venv/bin/python3
import rospy
from custom_msgs.srv import ChatPrompt, ChatPromptResponse
import openai

class ChatGPTServer:
    def __init__(self):
        # Initialize node
        rospy.init_node('chat_gpt_server', anonymous=True)

        # Set OpenAI API key (secure handling recommended)
        # self.api_key = 'sk-proj-h03D2Z0whXrmGLVb6DsATOyCsm7I_6jOGDZVY7CbjkTG3uR-CgFhTkghfojgXxk-wQCHpC13oVzT3BlbkFJH9Y_ZtbeZO1E2T6AQXaWuOliq8tN2GzbHtf_DcfhrCB_ET9ITgOyMsIYu4Fuj0T2XJQvntLwA'  # Replace with secure method (e.g., environment variable)
        
        # self.api_key = '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'  # Replace with secure method (e.g., environment variable)
        self.api_key = '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'  # Replace with secure method (e.g., environment variable)
        
        openai.api_key = self.api_key
        # openai.api_base = "https://api.groq.com/openai/v1"

        # Create ROS 1 service
        self.srv = rospy.Service('chat_gpt_ask', ChatPrompt, self.service_callback)

    def service_callback(self, req):
        # rospy.loginfo(f'Received prompt: {req.prompt}')
        response = self.get_chat_gpt_response(req.prompt)
        return ChatPromptResponse(response=response)

    def get_chat_gpt_response(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4o',
                # model='llama-3.3-70b-versatile',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )
            chat_response = response.choices[0].message.content
            rospy.loginfo(f'ChatGPT response: {chat_response}')
            return chat_response
        except Exception as e:
            rospy.logerr(f'Error calling OpenAI API: {e}')
            return "Error retrieving response from ChatGPT."

def main():
    server = ChatGPTServer()
    rospy.loginfo("ChatGPT server started.")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
