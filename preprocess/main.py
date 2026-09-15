import asyncio
import os

from agents import PIAgent, GEOAgent, TCGAAgent, FilterAgent, CodeReviewerAgent, DomainExpertAgent
from core.au_generator import generate_and_save_all
from core.context import ActionUnit
from core.prompt_loader import load_prompts, load_action_units
from environment import Environment
from utils.config import setup_arg_parser
from utils.llm import get_llm_client, get_role_specific_args
from utils.logger import Logger
from utils.utils import extract_function_code, get_question_pairs, check_slow_inference


async def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    model = args.model
    scaler = 1.0
    if check_slow_inference(model, args.thinking):
        scaler = 6.0 if ('deepseek' in model.lower() and '671b' in model.lower()) else 3.0
    elif ('deepseek' in model.lower() and 'v3' in model.lower()):
        scaler = 3.0
    args.max_time = args.max_time * scaler
    task_info_file = './metadata/task_info.json'
    all_pairs = get_question_pairs(task_info_file)
    if args.max_traits:
        seen_traits = set()
        filtered_pairs = []
        for pair in all_pairs:
            if pair[0] not in seen_traits:
                if len(seen_traits) >= args.max_traits:
                    break
                seen_traits.add(pair[0])
            filtered_pairs.append(pair)
        all_pairs = filtered_pairs
    in_data_root = args.data_root
    
    tcga_root = os.path.join(in_data_root, 'TCGA')
    output_root = './output/'
    version = args.version
    log_file = os.path.join(output_root, f"log_{version}.txt")
    
    logger = Logger(log_file=log_file, max_msg_length=getattr(args, 'max_log_msg_chars', 10000))
    
    # Role-specific clients
    pi_client = get_llm_client(get_role_specific_args(args, 'pi'), logger)
    data_engineer_client = get_llm_client(get_role_specific_args(args, 'data-engineer'), logger)
    filter_client = get_llm_client(get_role_specific_args(args, 'filter'), logger)
    code_reviewer_client = get_llm_client(get_role_specific_args(args, 'code-reviewer'), logger)
    domain_expert_client = get_llm_client(get_role_specific_args(args, 'domain-expert'), logger)
    planning_client = get_llm_client(get_role_specific_args(args, 'planning'), logger)
    
    # Handle generation of action units if requested
    use_generated = False
    if args.generate_action_units:
        logger.info("Generating Action Unit prompts from guidelines. This may take a few minutes...")
        
        # Generate and save prompts
        generated_files = await generate_and_save_all(
            planning_client, 
            logger,
        )
        
        if generated_files:
            logger.info("\nGenerated Action Unit prompt files:")
            for agent_id, filepath in generated_files.items():
                logger.info(f"  {agent_id}: {filepath}")
            
            # Human-in-the-loop refinement
            if not args.non_interactive:
                print("\n" + "="*60)
                print("Action Unit prompts have been generated.")
                print("You can now edit the files listed above to refine them.")
                print("Press Enter when you're done editing...")
                print("="*60)
                input()

            # Confirmation
            if args.non_interactive:
                # In non-interactive mode, skip confirmation and use generated prompts by default
                use_generated = True
                logger.info("Non-interactive mode: auto-confirmed use of generated Action Unit prompts")
            else:
                print("\nDo you want to use the generated prompts?")
                confirm = input("[Y/n]: ").strip().lower() or 'y'
                
                if confirm == 'y':
                    use_generated = True
                    logger.info("Using generated Action Unit prompts")
                else:
                    logger.info("Using base Action Unit prompts (ignoring generated files)")
        else:
            logger.warning("Failed to generate any prompts, using base prompts")
    
    # Load prompts (role, guidelines, tools, etc.)
    prompts = load_prompts()

    prep_tool_file = "./tools/preprocess.py"
    with open(prep_tool_file, 'r') as file:
        prep_tools_code_full = file.read()
    geo_selected_code = extract_function_code(prep_tool_file,
                                              ["validate_and_save_cohort_info", "geo_select_clinical_features",
                                               "preview_df"])
    geo_tools = {"full": prompts.PREPROCESS_TOOLS.format(tools_code=prep_tools_code_full),
                 "domain_focus": prompts.PREPROCESS_TOOLS.format(tools_code=geo_selected_code)}
    
    # Load action units for GEO agent
    geo_action_units = load_action_units('geo', use_generated=use_generated)

    geo_agent = GEOAgent(
        client=data_engineer_client,
        logger=logger,
        role_prompt=prompts.GEO_ROLE_PROMPT,
        guidelines=prompts.GEO_GUIDELINES,
        tools=geo_tools,
        setups='',
        action_units=geo_action_units,
        args=args,
        planning_client=planning_client
    )

    tcga_selected_code = extract_function_code(prep_tool_file, ["tcga_get_relevant_filepaths",
                                                                "tcga_convert_trait",
                                                                "tcga_convert_age",
                                                                "tcga_convert_gender",
                                                                "tcga_select_clinical_features",
                                                                "preview_df"])
    tcga_tools = {"full": prompts.PREPROCESS_TOOLS.format(tools_code=prep_tools_code_full),
                  "domain_focus": prompts.PREPROCESS_TOOLS.format(tools_code=tcga_selected_code)}
    
    # Load action units for TCGA agent with subdirectory context
    tcga_action_units = load_action_units('tcga', use_generated=use_generated, 
                                          context={'subdirs': os.listdir(tcga_root)})

    tcga_agent = TCGAAgent(
        client=data_engineer_client,
        logger=logger,
        role_prompt=prompts.TCGA_ROLE_PROMPT,
        guidelines=prompts.TCGA_GUIDELINES,
        tools=tcga_tools,
        setups='',
        action_units=tcga_action_units,
        args=args,
        planning_client=planning_client
    )


    # Set up Filter agent
    filter_tool_file = "./tools/filter.py"
    with open(filter_tool_file, 'r') as file:
        filter_tools_code_full = file.read()
    filter_tools = {"full": prompts.FILTER_TOOLS.format(tools_code=filter_tools_code_full),
                    "domain_focus": prompts.FILTER_TOOLS.format(tools_code=filter_tools_code_full)}

    filter_action_units = load_action_units('filter', use_generated=use_generated)

    filter_agent = FilterAgent(
        client=filter_client,
        logger=logger,
        role_prompt=prompts.FILTER_ROLE_PROMPT,
        guidelines=prompts.FILTER_GUIDELINES,
        tools=filter_tools,
        setups='',
        action_units=filter_action_units,
        args=args,
        planning_client=planning_client
    )

    agents = [
        PIAgent(client=pi_client, logger=logger, args=args),
        geo_agent,
        tcga_agent,
        filter_agent,
        CodeReviewerAgent(client=code_reviewer_client, logger=logger, args=args),
        DomainExpertAgent(client=domain_expert_client, logger=logger, args=args)
    ]

    env = Environment(logger=logger, agents=agents, args=args)

    await env.run(all_pairs, in_data_root, output_root, version)


if __name__ == "__main__":
    asyncio.run(main())
