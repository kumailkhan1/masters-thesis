export default interface RetrievedNode {
    metadata: {
        Title: string;
        DOI: string;
        Authors: string;
        text: string;
    };
    score: number;
}