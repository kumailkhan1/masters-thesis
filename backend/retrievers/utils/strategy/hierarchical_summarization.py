import asyncio


async def acombine_results(
    texts,
    query_str,
    qa_prompt,
    llm,
    cur_prompt_list,
    num_children=10,
):
    try:
        
        fmt_prompts = []
        for idx in range(0, len(texts), num_children):
            text_batch = texts[idx : idx + num_children]
            context_str = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            fmt_prompts.append(fmt_qa_prompt)
            cur_prompt_list.append(fmt_qa_prompt)

        tasks = [llm.acomplete(p) for p in fmt_prompts]
        combined_responses = await asyncio.gather(*tasks)
        new_texts = [str(r) for r in combined_responses]

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return await acombine_results(
                new_texts, query_str, qa_prompt, llm, cur_prompt_list, num_children=num_children
            )
    except Exception as e:
        print(f"Error occurred: {e}")

async def agenerate_response_hs(
    retrieved_nodes, query_str, qa_prompt, llm, num_children=10
):
    """Generate a response using hierarchical summarization strategy.

    Combine num_children nodes hierarchically until we get one root node.

    """
    
    try:
        
        fmt_prompts = []
        node_responses = []
        for node in retrieved_nodes:
            context_str = node.get_content()
            fmt_qa_prompt = qa_prompt.format(
                context_str=context_str, query_str=query_str
            )
            fmt_prompts.append(fmt_qa_prompt)
        print(fmt_prompts)
        tasks = [llm.acomplete(p) for p in fmt_prompts]
        node_responses = await asyncio.gather(*tasks)

        response_txt = await acombine_results(
            [str(r) for r in node_responses],
            query_str,
            qa_prompt,
            llm,
            fmt_prompts,
            num_children=num_children,
        )

        return response_txt, fmt_prompts
    except Exception as e:
        print(f"Error occurred: {e}")