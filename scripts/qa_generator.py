import re
import random
import json
import os

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

def GPT_API(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    output = response.choices[0].message.content
    return output

QUESTION_TOKEN = "<question>"

class QAGenerator:
    def __init__(self, task_list, q2_task_list=None, max_retries=2):
        self.task_list = task_list
        self.q2_task_list = q2_task_list if q2_task_list is not None else ["Video Caption", "Task Planning", "Action Temporal Localization", "Action Segment Summarization", "Action Segmentation and Summarization"]
        self.pretrain_stage_name = ['Pretrain', 'pretrain']
        self.finetune_stage_name = ['Finetune', 'finetune']
        self.max_retries = max_retries
        self.video_token = "<image>"

    def generate_better_caption(self, caption, task):
        '''pretrain stage'''
        if task not in ['fractal20220817_data', 'libero_spatial_no_noops', 'libero_goal_no_noops', 'libero_10_no_noops', 'droid', 'bc_z', 'robo_set', 'utokyo_xarm_bimanual_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds',
                        'calvin', 'franka_kitchen', 'bridge_data_v2_combine', 'bridge_data_v2_combine_rss']:
          prompt = f"Complete the phrase {caption} into a full sentence within the context of a robot performing a tabletop manipulation task. Only add the subject, verb, and object; no extra details are needed. If a coherent sentence cannot be generated, return -1."
          output = GPT_API(prompt=prompt)
        else:
          output = f"The robot {caption}"
        return output
    
    def get_qa_prompt(self, Q_type, raw_data):
        '''
        finetune stage
        now support task
        Action Identification, Object Identification, Spatial Relationship, Action Ordering, Temporal Localization, Task Success Detection,
        Video Caption, 
        '''
        if Q_type not in self.q2_task_list:
            template = "Please ask a question for {Q_type} in the field of video understanding based on the video. Then you need to give the answer.\n Video Content: {input}\n # The video depicts a robot performing tasks on a tabletop.\n # Only describe what you are certain about, and avoid providing descriptions that may be ambiguous or inaccurate.\n # Your response must be in JSON format, with the keys question and answer, like this:\n {{question: your question here, answer: your answer here}}."
            if Q_type not in ["Task Success Detection"]:
                raw_type_list = ["step_instructions", "frame_segment", "total_frames"]
            else:
                raw_type_list = ["step_instructions", "frame_segment", "total_frames", "current_frame"]
            raw_information = {raw_type: raw_data[raw_type] for raw_type in raw_type_list}
            prompt = template.format(Q_type=Q_type, input=raw_information)

        elif Q_type == "Video Caption":
            template = "Complete the phrase {input} into a full sentence within the context of a robot performing a tabletop manipulation task. Only add the subject, verb, and object; no extra details are needed. If a coherent sentence cannot be generated, return -1."
            prompt = template.format(input=raw_data["step_instructions"])

        elif Q_type == "Action Temporal Localization":
            step_idx = random.randint(0, len(raw_data["frame_segment"])-1)
            action_segment = raw_data["temporal_segment"][step_idx]
            rounded_segment = [round(t, 2) for t in action_segment]
            # segment format: frame {start_frame_index} to {end_frame_index}.
            action_description = raw_data["step_instructions"][step_idx]
            prompt = [action_description, f't={rounded_segment[0]} to t={rounded_segment[1]}.']
        
        elif Q_type == "Action Segment Summarization":
            step_idx = random.randint(0, len(raw_data["frame_segment"])-1)
            action_segment = raw_data["temporal_segment"][step_idx]
            rounded_segment = [round(t, 2) for t in action_segment]
            action_description = raw_data["step_instructions"][step_idx]
            prompt = [rounded_segment, action_description]

        elif Q_type == "Action Segmentation and Summarization":
            # Frames {start_frame_1}-{end_frame_1}: [Brief description of action 1]
            prompt = []
            for step_idx in range(len(raw_data["frame_segment"])):
                action_segment = raw_data["temporal_segment"][step_idx]
                rounded_segment = [round(t, 2) for t in action_segment]
                action_description = raw_data["step_instructions"][step_idx]
                description = f"t={rounded_segment[0]} to t={rounded_segment[1]}: {action_description}"
                prompt.append(f"{step_idx+1}. {description}")
            prompt = "\n".join(prompt)

        elif Q_type == "Task Planning":
            template = "Complete the phrase {input} into a full sentence within the context of a robot performing a tabletop manipulation task. Only add the subject, verb, and object; no extra details are needed. If a coherent sentence cannot be generated, return -1."
            task_prompt =  template.format(input=raw_data["step_instructions"])
            template = "You are directing a robot to perform a tabletop manipulation task.\n What's the next step action decision you need to make based on the video?\n You need to give the answer directly.\n Video Content: {input}\n # The video depicts a robot performing tasks on a tabletop.\n # Only describe what you are certain about, and avoid providing descriptions that may be ambiguous or inaccurate.\n"
            raw_type_list = ["step_instructions", "frame_segment", "total_frames", "current_frame"]
            raw_information = {raw_type: raw_data[raw_type] for raw_type in raw_type_list}
            prompt = [task_prompt, template.format(input=raw_information)]
        
        return prompt
        
    
    def get_question(self, Q_type):
        if Q_type == "Video Caption":
            question_template = [
                "Describe the video briefly.",
                "Summarize the video's content.",
                "Explain the video's main idea.",
                "Provide a brief overview of the video.",
                "Highlight the key points in the video.",
                "Summarize what happens in the video.",
                "Give a short description of the video's subject.",
                "Outline the video's main events.",
                "Describe the video's plot.",
                "Provide a concise summary of the video.",
                "Summarize the main events shown in the video.",
                "Give an overview of the video's key points.",
                "Describe the video's content succinctly.",
                "Briefly describe what the video is about.",
                "Provide a short description of the video's content.",
                "Explain the content of the video in a few words.",
                "Give a brief description of the video's subject matter.",
                "Summarize the key events in the video.",
                "Highlight the main events shown in the video.",
                "Provide a rundown of the video's primary events.",
                "Outline the significant events in the video.",
                "Sketch the main happenings in the video.",
                "Give an overview of the video's content.",
                "Provide a high-level summary of the video's content.",
                "Offer a synopsis of the video's content.",
                "Share a broad overview of the video's content.",
                "Summarize the content of this video briefly.",
                "Share a brief of the video's captured moments.",
                "Provide a quick summary of what is captured in the video.",
                "Describe the captured content in the video briefly.",
                "Offer a short overview of what the video captures.",
                "Summarize what is depicted in the video.",
                "Describe the video's main themes.",
                "Outline the video's important points.",
                "Provide an in-depth summary of the video.",
                "Explain the main topics covered in the video.",
                "Give a detailed description of the video's content.",
                "Summarize the video's important moments.",
                "Provide a comprehensive overview of the video.",
                "Explain the video's key elements.",
                "Outline the video's main discussions.",
                "Summarize the main messages in the video.",
                "Describe the essential points of the video.",
                "Provide a detailed overview of the video's subject.",
                "Give a thorough explanation of the video's content.",
                "Summarize the primary themes in the video.",
                "Outline the video's major ideas.",
                "Provide a full summary of the video's content.",
                "Explain the significant aspects of the video.",
                "Describe the critical elements in the video."
            ]

        elif Q_type == "Task Planning":
            question_template = [
                "You are directing a robot to perform tasks on a tabletop. Given the historical video observation and the instruction to \"{task_instruction}\", what's the next step action decision you need to make?",
                "As you guide a robot to execute tasks on a tabletop, and considering the prior video observation, what is the next action you need to direct after receiving the instruction to \"{task_instruction}\"?",
                "Based on the robot's recorded activities and the command \"{task_instruction}\", what is the next action you should instruct on the tabletop?",
                "Given the prior actions observed in the video and the directive \"{task_instruction}\", what is your next step in guiding the robot on the tabletop?",
                "Reflecting on the robot's past performance and the command to \"{task_instruction}\", what should your next action decision be on the tabletop?",
                "With the robot's prior video observation and the instruction to \"{task_instruction}\" in mind, what action should you instruct next?",
                "Considering the robot's previous video history and the task \"{task_instruction}\", what is the appropriate next action to direct?",
                "Considering the previous video observations, what would be your next move after being instructed to \"{task_instruction}\"?",
                "With the current instruction to \"{task_instruction}\", and the historical video data at hand, what is your next step in guiding the robot?",
                "Given the instruction \"{task_instruction}\" and the past video context, what is the next task you would execute?",
                "Looking at the past video footage, how would you proceed after receiving the instruction \"{task_instruction}\"?",
                "Taking into account the instruction \"{task_instruction}\" and earlier video observations, what is the next action step?",
                "In light of the previous video context, what is your next course of action with the instruction \"{task_instruction}\"?",
                "Using the instruction \"{task_instruction}\" and the recorded video data, what would be the next logical step in task planning?"
            ]

        elif Q_type == "Action Temporal Localization":
            question_template = [
            "Where in the video can you locate the segment that corresponds to the action '{action_description}'?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "During which period in the video does the action '{action_description}' occur?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "Find the specific time range in the video where the robot is executing '{action_description}'.\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "Can you identify the time in the video that matches the action '{action_description}'?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "Which part of the video shows the robot performing the action '{action_description}'?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "Locate the exact moment in the video when the robot carries out '{action_description}'.\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "Where in the video timeline does the action '{action_description}' take place?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "At what point can the action '{action_description}' be observed in the video?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "Identify when in the video the robot performs '{action_description}'.\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "During which section of the video does the robot execute the action '{action_description}'?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "In which part of the video is the robot seen performing '{action_description}'?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "At what point in the video timeline can you observe the robot completing '{action_description}'?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "Which time segment of the video shows the robot engaged in '{action_description}'?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "In what time frame does the action '{action_description}' unfold in the video?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "During which exact time range does the action '{action_description}' occur, from start to end?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "What is the time span for the action '{action_description}' in the video, including both start and end times?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "In which time interval does the action '{action_description}' start and conclude in the video?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "What is the full time range, from start to end, of the action '{action_description}' in the video?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "At what points in the video does the action '{action_description}' start and finish?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1.",
            "Can you specify when the action '{action_description}' begins and ends within the video?\n# Your answer must be in this format: t=t_start to t=t_end, where t_start and t_end are normalized time coordinates between 0 and 1."
        ]
        
        elif Q_type == "Action Segment Summarization":
            question_template = [
            "What is a concise summary of the action shown in the video segment from t={t_start} to t={t_end}?\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Provide a brief description of the key action occurring between t={t_start} and t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Summarize the main activity depicted in the video clip spanning from t={t_start} to t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "In a few words, what is the primary action taking place from t={t_start} to t={t_end}?\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Describe succinctly the core action presented in the video segment (t={t_start} to t={t_end}).\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "What is the essence of the action captured in the video clip between t={t_start} and t={t_end}?\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Offer a concise explanation of the main action shown from t={t_start} to t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Briefly characterize the central activity in the video segment spanning t={t_start} to t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "What is a short, precise description of the action occurring from t={t_start} to t={t_end}?\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Summarize in a few words the key action displayed in the video clip from t={t_start} to t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "What is the main action unfolding in the video between t={t_start} and t={t_end}?\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Briefly outline the key activity observed from t={t_start} to t={t_end} in the video.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "In a concise manner, describe the central action taking place during t={t_start} to t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Provide a short summary of the primary event occurring in the video segment (t={t_start} to t={t_end}).\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "What is a precise, brief description of the action shown from t={t_start} to t={t_end}?\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Summarize the core activity presented in the video clip spanning t={t_start} to t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "In a few words, what is the essential action captured between t={t_start} and t={t_end}?\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Concisely explain the main event depicted in the video portion from t={t_start} to t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Offer a brief characterization of the primary action seen from t={t_start} to t={t_end}.\n# Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "What is a succinct description of the key activity shown in the video segment (t={t_start} to t={t_end})?\n# Note: t_start and t_end are normalized time coordinates between 0 and 1."
        ]
    
        elif Q_type == "Action Segmentation and Summarization":
            question_template = [
            "Identify and briefly describe the key robotic actions in this video, specifying the start and end times for each action.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Segment the video into distinct robotic operations, providing a concise description and time range for each.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Outline the main robotic tasks performed in the video, including their temporal boundaries and brief summaries.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Describe the sequence of robotic actions observed, noting the starting and ending times for each identified action.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Summarize the critical robotic movements in the video, indicating the time intervals for each significant action.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Break down the video into separate robotic actions, offering a short description and timestamp for each segment.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Identify the primary robotic operations in the footage, specifying their durations and providing succinct explanations.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Analyze the video to extract key robotic maneuvers, detailing their time ranges and basic characteristics.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Partition the video into discrete robotic tasks, briefly explaining each and noting their respective time spans.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Enumerate the distinct robotic actions visible in the video, including their time frames and concise descriptions.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Segment the video based on distinct robotic actions, providing short descriptions and temporal boundaries.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Outline the sequence of robotic operations, noting the time intervals and offering brief explanations for each.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Break down the video into key robotic maneuvers, specifying their durations and providing concise summaries.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Analyze the footage to identify separate robotic tasks, including their timestamps and brief descriptions.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1.",
            "Summarize the primary robotic actions observed in the video, indicating their time ranges and key characteristics.\n# Your answer must be in this format: t=t_start to t=t_end: Brief description of action. Note: t_start and t_end are normalized time coordinates between 0 and 1."
        ]

        question = random.choice(question_template)
        return question

    def generate_gpt_qa(self, prompt, Q_type):
        def get_json_qa(prompt):
            output = GPT_API(prompt)
            try:
                match = re.search(r'{.*}', output, re.DOTALL)
                if not match:
                    raise ValueError("No match found")
            except Exception as e:
                print(f"An error occurred: {e}")
                output = GPT_API(prompt)
                match = re.search(r'{.*}', output, re.DOTALL)

            output = match.group()
            cleaned_output = output.replace('\x08', '')
            return cleaned_output
        def convert_keys(d):
            if isinstance(d, dict):
                return {k.strip().lower(): convert_keys(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_keys(i) for i in d]
            else:
                return d
            
        if Q_type in self.q2_task_list:
            question = self.get_question(Q_type=Q_type)

            if Q_type == "Task Planning":
                task_instruction = GPT_API(prompt[0])
                question = question.format(task_instruction=task_instruction)
                answer = GPT_API(prompt[1])
            if Q_type == "Action Temporal Localization":
                action_description = prompt[0]
                question = question.format(action_description=action_description)
                answer = prompt[1]
            elif Q_type == "Action Segment Summarization":
                action_segment = prompt[0]
                question = question.format(t_start=action_segment[0], t_end=action_segment[1])
                answer = prompt[1]
            elif Q_type == "Action Segmentation and Summarization":
                answer = prompt
            else:
                answer = GPT_API(prompt)
            qa_pair = {"question": question, "answer": answer}
        else:
            retries = 0
            while retries < self.max_retries:
                cleaned_output = get_json_qa(prompt)
                try:
                    raw_answer = json.loads(cleaned_output)
                    output = convert_keys(raw_answer) # lower + strip
                    output_keys = output.keys()
                    if 'question' in output_keys and 'answer'in output_keys:
                        qa_pair = output
                        break
                    else:
                        print('wrong keys', output_keys)
                        retries += 1
                except json.decoder.JSONDecodeError as e:
                    print(cleaned_output)
                    retries += 1
            if retries >= self.max_retries:
                qa_pair = -1

        return qa_pair

    def get_instance_template(self):
        instance = {
            "video": "video_name.mp4", 
            "conversations": [
            {
                "from": "human",
                "value": "question"
            },
            {
                "from": "gpt",
                "value": "answer"
            }
        ]}
        return instance
    
    def format_question_with_video(self, question=None):
        video_token = self.video_token
        formatted_question = f"{video_token}\n{question}" if question is not None else f"{video_token}\n"
        return formatted_question
    
    def generate_qa_instance(self, annotation, video_name, stage, task):
        if stage in self.pretrain_stage_name:
            qa_instance = self.get_instance_template()
            question = self.get_question(Q_type="Video Caption")
            step_instructions = annotation['step_instructions']
            cleaned_instructions = [step.rstrip('.,') for step in step_instructions]
            instructions = ', '.join(cleaned_instructions)
            if len(step_instructions) > 1:
              parts = instructions.rsplit(', ', 1)
              instructions = ' and '.join(parts)
            caption = self.generate_better_caption(caption=instructions, task=task)
            qa_instance['video'] = video_name
            qa_instance['conversations'][0]['value'] = self.format_question_with_video(question=question)
            qa_instance['conversations'][1]['value'] = caption

        elif stage in self.finetune_stage_name:
            qa_instance = []
            for q_type in self.task_list:
                prompt = self.get_qa_prompt(Q_type=q_type, raw_data=annotation)
                qa_pair = self.generate_gpt_qa(prompt=prompt, Q_type=q_type)
                if qa_pair != -1:
                    instance = self.get_instance_template()
                    instance['video'] = video_name
                    instance['conversations'][0]['value'] = self.format_question_with_video(question=qa_pair['question'])
                    instance['conversations'][1]['value'] = qa_pair['answer']
                    instance['question_type'] = q_type
                    qa_instance.append(instance)

        return qa_instance