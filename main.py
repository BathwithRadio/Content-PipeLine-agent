from typing import List
from crewai.flow.flow import Flow, listen, start, router, and_, or_
from crewai import Agent
from crewai import LLM # AI와 직접 대화하게 해주는 directory
from pydantic import BaseModel
from tools import web_search_tool

#role backstory goal tool을 써서 agent를 사용하지 않고도
#AI에게 요청을 보내고, 우리가 원하는 방식으로 Output을 내도록 할 것
class BlogPost(BaseModel):
    title: str
    subtitle: str
    sections: List[str]


class Tweet(BaseModel):
    content: str
    hashtags: str


class LinkedInPost(BaseModel):
    hook: str
    content: str
    call_to_action: str

class Score(BaseModel):
    score: int = 0
    reason: str = ""

class ContentPipelinState(BaseModel):
    
    #input
    content_type:str = ""
    topic: str = ""
    
    #Internal
    max_length: int = 0
    score: int = 0
    research: str = ""
    score: Score | None = None
    
    #Content
    blog_post:BlogPost | None = None
    tweet: str = ""
    linkedin_post:str = ""

class ContentPipelineFlow(Flow[ContentPipelinState]):
    
    #check pipiline validate state properly
    @start()
    def init_content_pipeline(self):
        if self.state.content_type not in ["tweet", "blog", "linkedin"]:
            raise ValueError("The content type is wrong")
        
        if self.state.topic == "":
            raise ValueError("The topic can't be blank")
        
        if self.state.content_type =="tweet":
            self.state.max_length = 150
        if self.state.content_type =="blog":
            self.state.max_length = 800
        if self.state.content_type =="linkedin":
            self.state.max_length = 500
            
    @listen(init_content_pipeline)
    def conduct_research(self):
        
        researcher = Agent(
            role="Head Researcher",
            backstory="You're like a digital detective who loves digging up fascinating facts and insights. You have a knack for finding the good stuff that others miss.",
            goal=f"Find the most interesting and useful info about {self.state.topic}",
            tools=[web_search_tool],
        )
        
        self.state.research = researcher.kickoff(
            f"Find the most interesting and useful info about {self.state.topic}",
        )
        
    @router(conduct_research)
    def conduct_research_router(self):
        content_type = self.state.content_type
        
        if content_type == "blog":
            return "make_blog"
        elif content_type == "tweet":
            return "make_tweet"
        else:
            return "make_linkedin_post"
    
    @listen(or_("make_blog", "remoke_blog"))
    def handle_make_blog(self):
        blog_post = self.state.blog_post
        
        llm = LLM(model="openai/o4-mini", response_format=BlogPost)
        
        
        if blog_post is None:
            self.state.blog_post = llm.call(f"""
                     Make a blog post on the topic {self.state.topic} using the following research:
                     <research>
                     =====================
                     {self.state.research}
                     =====================
                     </research>
                     """)
        else: 
            self.state.blog_post = llm.call(f"""
                     Improve this blog post on {self.state.topic}, but it does not have a good SEO score.
                     because of {self.state.score.reason}
                     Improve it.
                     <blog post>
                     {self.state.blog_post.model_dump_json()}
                     </blog post>
                        
                        Use the gollowing research.
                        
                     <research>
                     =====================
                     {self.state.research}
                     =====================
                     </research>
                     """)

    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        # if blog tweet has been made, show the old one to the ai and ask it to improve
        # else just ask to create
        print("Making blog tweet...")

    @listen(or_("make_linkedin_post", "remake_linkedin_post"))
    def handle_make_linkedin_post(self):
        # if linkedin post has been made, show the old one to the ai and ask it to improve
        # else just ask to create
        print("Making blog linkedin post...")
    
    @listen(handle_make_blog)
    def check_seo(self):
        print(self.state.blog_post)
        print("=================")
        print(self.state.research)
        print("Check Blog SEO...")
    
    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        print("Checking virality...")
    
    @router(or_(check_seo, check_virality))
    def score_router(self):
        content_type = self.state.content_type
        score = self.state.score
        
        if score >= 8:
            return "check_passed"
        else:
            if content_type == "blog":
                return "remake_blog"
            elif content_type == "linkedin":
                return "remake_linkedin_post"
            else:
                return "remake_tweet"
    
    @listen("check_passed")
    def finalize_content(self):
        print("Finalizing content")
        
        
flow = ContentPipelineFlow()

# flow.plot()
flow.kickoff(
    inputs={
        "content_type": "blog",
        "topic": "AI Dog Traning",
    },
             
)
