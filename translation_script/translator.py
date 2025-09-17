import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class TranslationWorker:
    def __init__(self, api_key, base_url, model, system_prompt, output_folder):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.output_folder = output_folder
        self.lock = threading.Lock()

    def translate_document(self, document, progress_callback=None):
        text_data = document["content"]
        text_title = document["title"]
        text_id = document["id"]

        user_input = f"""Translate the following Daoist text from Chinese into English:

<original_text>
{text_title}

{text_data}
</original_text>

Remember your task is to Translate classical Chinese Daoist texts into English in the style of James Legge's Zhuangzi examples above. Always provide a full translation. Always preserve markdown structure (headings, links, image links, lists, blockquotes). The output should have the following structure:
```markdown
# [Translated title of the scripture]

[Translated content of the scripture, complete markdown formatting(headings, links, image links, lists, blockquotes)]
```"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input},
                ],
                stream=True,
                temperature=0.72,
                top_p=1.0
            )

            complete_response = ""
            for chunk in response:
                if chunk.choices[0].finish_reason is not None:
                    break
                delta = chunk.choices[0].delta.content
                if delta:
                    complete_response += delta

            self.save_files(text_id, text_title, text_data, complete_response)

            if progress_callback:
                progress_callback(text_id)

            return {
                "id": text_id,
                "input": user_input,
                "output": complete_response
            }

        except Exception as e:
            print(f"Error processing {text_id}: {e}")
            return None

    def save_files(self, text_id, text_title, text_data, translation):
        os.makedirs(self.output_folder, exist_ok=True)

        original_path = os.path.join(self.output_folder, f"{text_id}.md")
        translated_path = os.path.join(self.output_folder, f"{text_id}_translated.md")

        with open(original_path, "w", encoding="utf-8") as f:
            f.write(f"# {text_title}\n\n{text_data}\n")

        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(translation)


class ParallelTranslator:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "openai/gpt-4.1"
        self.output_folder = "./output/"
        self.system_prompt = """You are a world-class translator of Daoist scriptures.  
Your task: Translate classical Chinese Daoist texts into English in the style of James Legge's Zhuangzi.  
- Always preserve markdown structure (headings, links, lists, blockquotes).  
- Never summarize or omit. Provide full literal translations.  
- Never add commentary, notes, or explanations — only the translation.  

## Style References: James Legge's Translation of Zhuangzi

### Example 1:

<example_input>
Title: 逍遙遊
Content: 北冥有魚，其名為鯤。鯤之大，不知其幾千里也。化而為鳥，其名為鵬。鵬之背，不知其幾千里也；怒而飛，其翼若垂天之雲。是鳥也，海運則將徙於南冥。南冥者，天池也。齊諧者，志怪者也。諧之言曰：「鵬之徙於南冥也，水擊三千里，摶扶搖而上者九萬里，去以六月息者也。」野馬也，塵埃也，生物之以息相吹也。天之蒼蒼，其正色邪？其遠而無所至極邪？其視下也亦若是，則已矣。且夫水之積也不厚，則負大舟也無力。覆杯水於坳堂之上，則芥為之舟，置杯焉則膠，水淺而舟大也。風之積也不厚，則其負大翼也無力。故九萬里則風斯在下矣，而後乃今培風；背負青天而莫之夭閼者，而後乃今將圖南。蜩與學鳩笑之曰：「我決起而飛，槍1榆、枋，時則不至而控於地而已矣，奚以之九萬里而南為？」適莽蒼者三湌而反，腹猶果然；適百里者宿舂糧；適千里者三月聚糧。之二蟲又何知！小知不及大知，小年不及大年。奚以知其然也？朝菌不知晦朔，蟪蛄不知春秋，此小年也。楚之南有冥靈者，以五百歲為春，五百歲為秋；上古有大椿者，以八千歲為春，八千歲為秋。而彭祖乃今以久特聞，眾人匹之，不亦悲乎！
</example_input>
<example_output>
Title: Enjoyment in Untroubled Ease
Content: In the Northern Ocean there is a fish, the name of which is Kun - I do not know how many li in size. It changes into a bird with the name of Peng, the back of which is (also) - I do not know how many li in extent. When this bird rouses itself and flies, its wings are like clouds all round the sky. When the sea is moved (so as to bear it along), it prepares to remove to the Southern Ocean. The Southern Ocean is the Pool of Heaven.
There is the (book called) Qi Xie, a record of marvels. We have in it these words: 'When the peng is removing to the Southern Ocean it flaps (its wings) on the water for 3000 li. Then it ascends on a whirlwind 90,000 li, and it rests only at the end of six months.' (But similar to this is the movement of the breezes which we call) the horses of the fields, of the dust (which quivers in the sunbeams), and of living things as they are blown against one another by the air. Is its azure the proper colour of the sky? Or is it occasioned by its distance and illimitable extent? If one were looking down (from above), the very same appearance would just meet his view.
And moreover, (to speak of) the accumulation of water; if it be not great, it will not have strength to support a large boat. Upset a cup of water in a cavity, and a straw will float on it as if it were a boat. Place a cup in it, and it will stick fast; the water is shallow and the boat is large. (So it is with) the accumulation of wind; if it be not great, it will not have strength to support great wings. Therefore (the peng ascended to) the height of 90,000 li, and there was such a mass of wind beneath it; thenceforth the accumulation of wind was sufficient. As it seemed to bear the blue sky on its back, and there was nothing to obstruct or arrest its course, it could pursue its way to the South.
A cicada and a little dove laughed at it, saying, 'We make an effort and fly towards an elm or sapanwood tree; and sometimes before we reach it, we can do no more but drop to the ground. Of what use is it for this (creature) to rise 90,000 li, and make for the South?' He who goes to the grassy suburbs, returning to the third meal (of the day), will have his belly as full as when he set out; he who goes to a distance of 100 li will have to pound his grain where he stops for the night; he who goes a thousand li, will have to carry with him provisions for three months. What should these two small creatures know about the matter? The knowledge of that which is small does not reach to that which is great; (the experience of) a few years does not reach to that of many. How do we know that it is so? The mushroom of a morning does not know (what takes place between) the beginning and end of a month; the short-lived cicada does not know (what takes place between) the spring and autumn. These are instances of a short term of life. In the south of Chu there is the (tree) called Ming-ling, whose spring is 500 years, and its autumn the same; in high antiquity there was that called Da-chun, whose spring was 8000 years, and its autumn the same. And Peng Zu is the one man renowned to the present day for his length of life: if all men were (to wish) to match him, would they not be miserable?
</example_output>

### Example 2:
<example_input>
Title: 人間世
Content: 顏回見仲尼請行。曰：「奚之？」曰：「將之衛。」曰：「奚為焉？」曰：「回聞衛君，其年壯，其行獨，輕用其國，而不見其過，輕用民死，死者以國量乎澤，若蕉，民其无如矣。回嘗聞之夫子曰：『治國去之，亂國就之，醫門多疾。』願以所聞思其則，庶幾其國有瘳乎！」仲尼曰：「譆！若殆往而刑耳！夫道不欲雜，雜則多，多則擾，擾則憂，憂而不救。古之至人，先存諸己，而後存諸人。所存於己者未定，何暇至於暴人之所行！且若亦知夫德之所蕩，而知之所為出乎哉？德蕩乎名，知出乎爭。名也者，相軋也；知也者，爭之器也。二者凶器，非所以盡行也。且德厚信矼，未達人氣；名聞不爭，未達人心。而彊以仁義繩墨之言術暴人之前者，是以人惡有其美也，命之曰菑人。菑人者，人必反菑之，若殆為人菑夫！且苟為悅賢而惡不肖，惡用而求有以異？若唯无詔，王公必將乘人而鬭其捷。而目將熒之，而色將平之，口將營之，容將形之，心且成之。是以火救火，以水救水，名之曰益多，順始无窮。若殆以不信厚言，必死於暴人之前矣。且昔者桀殺關龍逢，紂殺王子比干，是皆脩其身以下傴拊人之民，以下拂其上者也，故其君因其脩以擠之。是好名者也。昔者堯攻叢枝、胥敖，禹攻有扈，國為虛厲，身為刑戮，其用兵不止，其求實无已。是皆求名、實者也，而獨不聞之乎？名、實者，聖人之所不能勝也，而況若乎！雖然，若必有以也，嘗以語我來！」顏回曰：「端而虛，勉而一，則可乎？」曰：「惡！惡可？夫以陽為充孔揚，采色不定，常人之所不違，因案人之所感，以求容與其心。名之曰日漸之德不成，而況大德乎！將執而不化，外合而內不訾，其庸詎可乎！」「然則我內直而外曲，成而上比。內直者，與天為徒。與天為徒者，知天子之與己皆天之所子，而獨以己言蘄乎而人善之，蘄乎而人不善之邪？若然者，人謂之童子，是之謂與天為徒。外曲者，與人之為徒也。擎、跽、曲拳，人臣之禮也，人皆為之，吾敢不為邪！為人之所為者，人亦无疵焉，是之謂與人為徒。成而上比者，與古為徒。其言雖教，讁之實也。古之有也，非吾有也。若然者，雖直不為病，是之謂與古為徒。若是，則可乎？」仲尼曰：「惡！惡可？大多政，法而不諜，雖固，亦无罪。雖然，止是耳矣，夫胡可以及化！猶師心者也。」
</example_input>
<example_output>
Title: Man in the World, Associated with other Men
Content: Yan Hui went to see Zhongni, and asked leave to take his departure. 'Where are you going to?' asked the Master. 'I will go to Wei' was the reply. 'And with what object?' 'I have heard that the ruler of Wei is in the vigour of his years, and consults none but himself as to his course. He deals with his state as if it were a light matter, and has no perception of his errors. He thinks lightly of his people's dying; the dead are lying all over the country as if no smaller space could contain them; on the plains and about the marshes, they are as thick as heaps of fuel. The people know not where to turn to. I have heard you, Master, say, "Leave the state that is well governed; go to the state where disorder prevails." At the door of a physician there are many who are ill. I wish through what I have heard (from you) to think out some methods (of dealing with Wei), if peradventure the evils of the state may be cured.'
Zhongni said, 'Alas! The risk is that you will go only to suffer in the punishment (of yourself)! The right method (in such a case) will not admit of any admixture. With such admixture, the one method will become many methods. Their multiplication will embarrass you. That embarrassment will make you anxious. However anxious you may be, you will not save (yourself). The perfect men of old first had (what they wanted to do) in themselves, and afterwards they found (the response to it) in others. If what they wanted in themselves was not fixed, what leisure had they to go and interfere with the proceedings of any tyrannous man?
Moreover, do you know how virtue is liable to be dissipated, and how wisdom proceeds to display itself? Virtue is dissipated in (the pursuit of) the name for it, and wisdom seeks to display itself in the striving with others. In the pursuit of the name men overthrow one another; wisdom becomes a weapon of contention. Both these things are instruments of evil, and should not be allowed to have free course in one's conduct. Supposing one's virtue to be great and his sincerity firm, if he do not comprehend the spirit of those (whom he wishes to influence); and supposing he is free from the disposition to strive for reputation, if he do not comprehend their minds;-- when in such a case he forcibly insists on benevolence and righteousness, setting them forth in the strongest and most direct language, before the tyrant, then he, hating (his reprover's) possession of those excellences, will put him down as doing him injury. He who injures others is sure to be injured by them in return. You indeed will hardly escape being injured by the man (to whom you go)!
Further, if perchance he takes pleasure in men of worth and hates those of an opposite character, what is the use of your seeking to make yourself out to be different (from such men about him)? Before you have begun to announce (your views), he, as king and ruler, will take advantage of you, and immediately contend with you for victory. Your eyes will be dazed and full of perplexity; you will try to look pleased with him; you will frame your words with care; your demeanour will be conformed to his; you will confirm him in his views. In this way you will be adding fire to fire, and water to water, increasing, as we may express it, the evils (which you deplore). To these signs of deferring to him at the first there will be no end. You will be in danger, seeing he does not believe you, of making your words more strong, and you are sure to die at the hands of such a tyrant.
And formerly Jie killed Guan Long-feng, and Zhou killed the prince Bi-gan. Both of these cultivated their persons, bending down in sympathy with the lower people to comfort them suffering (as they did) from their oppressors, and on their account opposing their superiors. On this account, because they so ordered their conduct, their rulers compassed their destruction - such regard had they for their own fame. (Again), Yao anciently attacked (the states of) Cong-qi and Xu-ao, and Yu attacked the ruler of Hu. Those states were left empty, and with no one to continue their population, the people being exterminated. They had engaged in war without ceasing; their craving for whatever they could get was insatiable. And this (ruler of Wei) is, like them, one who craves after fame and greater substance - have you not heard it? Those sages were not able to overcome the thirst for fame and substance - how much less will you be able to do so! Nevertheless you must have some ground (for the course which you wish to take); pray try and tell it to me.'
Yan Hui said, 'May I go, doing so in uprightness and humility, using also every endeavour to be uniform (in my plans of operation)?' 'No, indeed!' was the reply. 'How can you do so? This man makes a display of being filled to overflowing (with virtue), and has great self-conceit. His feelings are not to be determined from his countenance. Ordinary men do not (venture to) oppose him, and he proceeds from the way in which he affects them to seek still more the satisfaction of his own mind. He may be described as unaffected by the (small lessons of) virtue brought to bear on him from day to day; and how much less will he be so by your great lessons? He will be obstinate, and refuse to be converted. He may outwardly agree with you, but inwardly there will be no self-condemnation - how can you (go to him in this way and be successful)?'
(Yan Hui) rejoined, 'Well then; while inwardly maintaining my straightforward intention, I will outwardly seem to bend to him. I will deliver (my lessons), and substantiate them by appealing to antiquity. Inwardly maintaining my straightforward intention, I shall be a co-worker with Heaven. When I thus speak of being a co-worker with Heaven, it is because I know that (the sovereign, whom we style) the son of Heaven, and myself, are equally regarded by Heaven as Its sons. And should I then, as if my words were only my own, be seeking to find whether men approved of them, or disapproved of them? In this way men will pronounce me a (sincere and simple) boy. This is what is called being a co-worker with Heaven. Outwardly bending (to the ruler), I shall be a co-worker with other men. To carry (the memorandum tablet to court), to kneel, and to bend the body reverentially - these are the observances of ministers. They all employ them, and should I presume not to do so? Doing what other men do, they would have no occasion to blame me. This is what is called being a fellow-worker with other men. Fully declaring my sentiments and substantiating them by appealing to antiquity, I shall be a co-worker with the ancients. Although the words in which I convey my lessons may really be condemnatory (of the ruler), they will be those of antiquity, and not my own. In this way, though straightforward, I shall be free from blame. This is what is called being a co-worker with antiquity. May I go to Wei in this way, and be successful?' 'No indeed!' said Zhongni. 'How can you do so? You have too many plans of proceeding, and have not spied out (the ruler's character). Though you firmly adhere to your plans, you may be held free from transgression, but this will be all the result. How can you (in this way) produce the transformation (which you desire)? All this only shows (in you) the mind of a teacher!'
</example_output>

### Example 3:
<example_input>
Title: 至樂
Content: 夫富者，苦身疾作，多積財而不得盡用，其為形也亦外矣。夫貴者，夜以繼日，思慮善否，其為形也亦疏矣。人之生也，與憂俱生，壽者惛惛，久憂不死，何苦也！其為形也亦遠矣。烈士為天下見善矣，未足以活身。吾未知善之誠善邪，誠不善邪？若以為善矣，不足活身；以為不善矣，足以活人。故曰：「忠諫不聽，蹲循勿爭。」故夫子胥爭之以殘其形，不爭，名亦不成。誠有善無有哉？今俗之所為與其所樂，吾又未知樂之果樂邪，果不樂邪？吾觀夫俗之所樂，舉群趣者，誙誙然如將不得已，而皆曰樂者，吾未之樂也，亦未之不樂也。果有樂無有哉？吾以無為誠樂矣，又俗之所大苦也。故曰：「至樂無樂，至譽無譽。」
</example_input>
<example_output>
Title: Perfect Enjoyment
Content: Now the rich embitter their lives by their incessant labours; they accumulate more wealth than they can use: while they act thus for the body, they make it external to themselves. Those who seek for honours carry their pursuit of them from the day into the night, full of anxiety about their methods whether they are skilful or not: while they act thus for the body they treat it as if it were indifferent to them. The birth of man is at the same time the birth of his sorrow; and if he live long he becomes more and more stupid, and the longer is his anxiety that he may not die; how great is his bitterness!-- while he thus acts for his body, it is for a distant result. Meritorious officers are regarded by the world as good; but (their goodness) is not sufficient to keep their persons alive. I do not know whether the goodness ascribed to them be really good or really not good. If indeed it be considered good, it is not sufficient to preserve their persons alive; if it be deemed not good, it is sufficient to preserve other men alive. Hence it is said, 'When faithful remonstrances are not listened to, (the remonstrant) should sit still, let (his ruler) take his course, and not strive with him.' Therefore when Zi-xu strove with (his ruler), he brought on himself the mutilation of his body. If he had not so striven, he would not have acquired his fame: was such (goodness) really good or was it not? As to what the common people now do, and what they find their enjoyment in, I do not know whether the enjoyment be really enjoyment or really not. I see them in their pursuit of it following after all their aims as if with the determination of death, and as if they could not stop in their course; but what they call enjoyment would not be so to me, while yet I do not say that there is no enjoyment in it. Is there indeed such enjoyment, or is there not? I consider doing nothing (to obtain it) to be the great enjoyment, while ordinarily people consider it to be a great evil. Hence it is said, 'Perfect enjoyment is to be without enjoyment; the highest praise is to be without praise.'
</example_output>"""
        self.lock = threading.Lock()
        self.completed_count = 0
        self.total_count = 0

    def get_pending_documents(self):
        with open("docs.json", "r", encoding="utf-8") as f:
            documents = json.load(f)

        completed = []
        if os.path.exists("output.json"):
            with open("output.json", "r", encoding="utf-8") as f:
                completed = json.load(f)

        completed_ids = {entry["id"] for entry in completed}
        pending = [doc for doc in documents if doc["id"] not in completed_ids]

        return pending, completed

    def progress_update(self, document_id):
        with self.lock:
            self.completed_count += 1
            print(f"Completed {document_id} - Progress: {self.completed_count}/{self.total_count}")

    def save_result(self, result, all_results):
        if result:
            with self.lock:
                all_results.append(result)
                temp_path = "output_tmp.json"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, "output_new.json")

    def translate_parallel(self):
        pending_documents, existing_results = self.get_pending_documents()

        if not pending_documents:
            print("No pending documents to translate.")
            return existing_results

        self.total_count = len(pending_documents)
        print(f"Starting translation of {self.total_count} documents with {self.max_workers} workers")

        all_results = existing_results.copy()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            workers = [
                TranslationWorker(
                    self.api_key,
                    self.base_url,
                    self.model,
                    self.system_prompt,
                    self.output_folder
                ) for _ in range(self.max_workers)
            ]

            futures = []
            for i, document in enumerate(pending_documents):
                worker = workers[i % self.max_workers]
                future = executor.submit(
                    worker.translate_document,
                    document,
                    self.progress_update
                )
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                self.save_result(result, all_results)

        print(f"Translation completed. Total documents processed: {len(all_results)}")
        return all_results


def main():
    translator = ParallelTranslator(max_workers=20)
    translator.translate_parallel()


if __name__ == "__main__":
    main()