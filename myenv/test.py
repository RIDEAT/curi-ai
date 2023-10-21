from langsmith import Client


client = Client()
dataset_name = "choosuk"

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


datasets = client.list_datasets(dataset_name=dataset_name)

# Since chains and agents can be stateful (they can have memory),
# create a constructor to pass in to the run_on_dataset method.
def create_chain():
    llm = ChatOpenAI(temperature=0)
    return LLMChain.from_string(llm, "Spit some bars about {input}.")

from langchain.smith import RunEvalConfig, run_on_dataset

eval_config = RunEvalConfig(
  evaluators=[
    # You can specify an evaluator by name/enum.
    # In this case, the default criterion is "helpfulness"
    "criteria",
    # Or you can configure the evaluator
    RunEvalConfig.Criteria("harmfulness"),
    RunEvalConfig.Criteria(
      {"cliche": "Are the lyrics cliche?"
      " Respond Y if they are, N if they're entirely unique."}
      )
  ]
)
run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=create_chain,
    evaluation=eval_config,
    verbose=True,
    project_name="llmchain-test-2",
)