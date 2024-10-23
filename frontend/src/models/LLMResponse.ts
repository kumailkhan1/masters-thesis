import RetrievedNode from "./RetrievedNode";

export default interface LLMResponse {
    response: string;
    retrieved_nodes: RetrievedNode[];
}