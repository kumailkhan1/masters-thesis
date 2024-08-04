

async def generate_response_cr(
    retrieved_nodes, query_str, qa_prompt, refine_prompt, llm
):
    cur_response = None
    fmt_prompts = []
    for idx, node in enumerate(retrieved_nodes):
        context_str = node.get_content()
        if idx == 0:
            fmt_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
        else:
            fmt_prompt = refine_prompt.format(
                context_str=context_str,
                query_str=query_str,
                existing_answer=str(cur_response),
            )

        cur_response = llm.complete(fmt_prompt)
        fmt_prompts.append(fmt_prompt)

    return cur_response, fmt_prompts