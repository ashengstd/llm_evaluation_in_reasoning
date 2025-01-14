import logging
import re

import rich.progress

from llm_evaluation_in_reasoning.dataloader.base import BaseBenchDataloader


class SimpleBench(BaseBenchDataloader):
    def __init__(self, progress: rich.progress.Progress) -> None:
        self.dataset = DATA["eval_data"]
        self.progress_bar = progress
        self.question_key = "prompt"
        self.answer_key = "answer"

    @staticmethod
    def extract_answer(output: str) -> str:
        try:
            output = output.strip()
            match = re.search(r"Final Answer:\s*([A-F])", output, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                logging.info(f"Answer extracted: {answer}")
                return answer
            else:
                logging.error("Answer not found")
                return "nan"
        except Exception as e:
            logging.error(f"Error extracting answer: {e}")
            return "nan"

    @staticmethod
    def eval_single_question(predicted_answer: str, answer: str) -> bool:
        logging.info(f"Predicted answer: {predicted_answer}, Ground truth: {answer}")
        return predicted_answer == answer

    @staticmethod
    def vote_majority(output: list[str], answer: str) -> bool:
        max_vote = max(set(output), key=output.count)
        logging.info(f"Output: {max_vote}, Ground truth: {answer}")
        return output.count(answer) > len(output) / 2

    def inital_default_prompt(self) -> str:
        return "You are a careful and systematic reasoning expert who excels at analyzing complex problems. For each question:\n\n1. Break down the problem into smaller components\n2. Evaluate each piece of evidence objectively\n3. Consider multiple perspectives and potential outcomes\n4. Assess the probability and practicality of each option\n5. Validate your reasoning process before concluding\n\nPresent your analysis in clear, logical steps. Support your reasoning with specific examples or evidence when possible. After your step-by-step analysis, provide your final answer in the following format:\n\nFinal Answer: X\n\nwhere X is one of the letters A, B, C, D, E, or F."


DATA = {
    "eval_data": [
        {
            "question_id": 1,
            "prompt": "Beth places four whole ice cubes in a frying pan at the start of the first minute, then five at the start of the second minute and some more at the start of the third minute, but none in the fourth minute. If the average number of ice cubes per minute placed in the pan while it was frying a crispy egg was five, how many whole ice cubes can be found in the pan at the end of the third minute?\nA. 30\nB. 0\nC. 20\nD. 10\nE. 11\nF. 5\n",
            "answer": "B",
        },
        {
            "question_id": 2,
            "prompt": "A juggler throws a solid blue ball a meter in the air and then a solid purple ball (of the same size) two meters in the air. She then climbs to the top of a tall ladder carefully, balancing a yellow balloon on her head. Where is the purple ball most likely now, in relation to the blue ball?\nA. at the same height as the blue ball\nB. at the same height as the yellow balloon\nC. inside the blue ball\nD. above the yellow balloon\nE. below the blue ball\nF. above the blue ball\n",
            "answer": "A",
        },
        {
            "question_id": 3,
            "prompt": "Jeff, Jo and Jim are in a 200m men's race, starting from the same position. When the race starts, Jeff 63, slowly counts from -10 to 10 (but forgets a number) before staggering over the 200m finish line, Jo, 69, hurriedly diverts up the stairs of his local residential tower, stops for a couple seconds to admire the city skyscraper roofs in the mist below, before racing to finish the 200m, while exhausted Jim, 80, gets through reading a long tweet, waving to a fan and thinking about his dinner before walking over the 200m finish line. [ _ ] likely finished last.\nA. Jo likely finished last\nB. Jeff and Jim likely finished last, at the same time\nC. Jim likely finished last\nD. Jeff likely finished last\nE. All of them finished simultaneously\nF. Jo and Jim likely finished last, at the same time\n",
            "answer": "A",
        },
        {
            "question_id": 4,
            "prompt": 'There are two sisters, Amy who always speaks mistruths and Sam who always lies. You don\'t know which is which. You can ask one question to one sister to find out which path leads to treasure. Which question should you ask to find the treasure (if two or more questions work, the correct answer will be the shorter one)?\nA. "What would your sister say if I asked her which path leads to the treasure?"\nB. "What is your sister\u2019s name?\u201d\nC. "What path leads to the treasure?"\nD. "What path do you think I will take, if you were to guess?"\nE. "What is in the treasure?"\nF. \u201cWhat is your sister\u2019s number?\u201d\n',
            "answer": "C",
        },
        {
            "question_id": 5,
            "prompt": "Peter needs CPR from his best friend Paul, the only person around. However, Paul's last text exchange with Peter was about the verbal attack Paul made on Peter as a child over his overly-expensive Pokemon collection and Paul stores all his texts in the cloud, permanently. Paul will [ _ ] help Peter.\nA. probably not\nB. definitely\nC. half-heartedly\nD. not\nE. pretend to\nF. ponder deeply over whether to\n",
            "answer": "B",
        },
        {
            "question_id": 6,
            "prompt": "While Jen was miles away from care-free John, she hooked-up with Jack, through Tinder. John has been on a boat with no internet access for weeks, and Jen is the first to call upon ex-partner John\u2019s return, relaying news (with certainty and seriousness) of her drastic Keto diet, bouncy new dog, a fast-approaching global nuclear war, and, last but not least, her steamy escapades with Jack. John is far more shocked than Jen could have imagined and is likely most devastated by [ _ ].\nA. wider international events\nB. the lack of internet\nC. the dog without prior agreement\nD. sea sickness\nE. the drastic diet\nF. the escapades\n",
            "answer": "A",
        },
        {
            "question_id": 7,
            "prompt": "John is 24 and a kind, thoughtful and apologetic person. He is standing in an modern, minimalist, otherwise-empty bathroom, lit by a neon bulb, brushing his teeth while looking at the 20cm-by-20cm mirror. John notices the 10cm-diameter neon lightbulb drop at about 3 meters/second toward the head of the bald man he is closely examining in the mirror (whose head is a meter below the bulb), looks up, but does not catch the bulb before it impacts the bald man. The bald man curses, yells 'what an idiot!' and leaves the bathroom. Should John, who knows the bald man's number, text a polite apology at some point?\nA. no, because the lightbulb was essentially unavoidable\nB. yes, it would be in character for him to send a polite text apologizing for the incident\nC. no, because it would be redundant\nD. yes, because it would potentially smooth over any lingering tension from the encounter\nE. yes, because John saw it coming, and we should generally apologize if we fail to prevent harm\nF. yes because it is the polite thing to do, even if it wasn't your fault.\n",
            "answer": "C",
        },
        {
            "question_id": 8,
            "prompt": "On a shelf, there is only a green apple, red pear, and pink peach. Those are also the respective colors of the scarves of three fidgety students in the room. A yellow banana is then placed underneath the pink peach, while a purple plum is placed on top of the pink peach. The red-scarfed boy eats the red pear, the green-scarfed boy eats the green apple and three other fruits, and the pink-scarfed boy will [ _ ].\nA. eat just the yellow banana\nB. eat the pink, yellow and purple fruits\nC. eat just the purple plum\nD. eat the pink peach\nE. eat two fruits\nF. eat no fruits\n",
            "answer": "F",
        },
        {
            "question_id": 9,
            "prompt": "Agatha makes a stack of 5 cold, fresh single-slice ham sandwiches (with no sauces or condiments) in Room A, then immediately uses duct tape to stick the top surface of the uppermost sandwich to the bottom of her walking stick. She then walks to Room B, with her walking stick, so how many whole sandwiches are there now, in each room?\nA. 4 whole sandwiches in room A, 0 whole sandwiches in Room B\nB. no sandwiches anywhere\nC. 4 whole sandwiches in room B, 1 whole sandwich in Room A\nD. All 5 whole sandwiches in Room B\nE. 4 whole sandwiches in Room B, 1 whole sandwiches in room A\nF. All 5 whole sandwiches in Room A\n",
            "answer": "A",
        },
        {
            "question_id": 10,
            "prompt": "A luxury sports-car is traveling north at 30km/h over a roadbridge, 250m long, which runs over a river that is flowing at 5km/h eastward. The wind is blowing at 1km/h westward, slow enough not to bother the pedestrians snapping photos of the car from both sides of the roadbridge as the car passes. A glove was stored in the trunk of the car, but slips out of a hole and drops out when the car is half-way over the bridge. Assume the car continues in the same direction at the same speed, and the wind and river continue to move as stated. 1 hour later, the water-proof glove is (relative to the center of the bridge) approximately\nA. 4km eastward\nB. <1 km northward\nC. >30km away north-westerly\nD. 30 km northward\nE. >30 km away north-easterly.\nF. 5 km+ eastward\n",
            "answer": "B",
        },
    ]
}
