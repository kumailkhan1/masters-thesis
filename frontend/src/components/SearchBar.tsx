import React, { useState } from 'react';
import axios from 'axios';
import { Oval } from 'react-loader-spinner';
import { RetrievedNodes } from '../models/RetrievedNodes';

interface LLMResponse {
    response: string;
    retrieved_nodes: RetrievedNodes[];
}



const SearchBar: React.FC = () => {
    const [query, setQuery] = useState<string>('');
    const [response, setResponse] = useState<LLMResponse | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');

    const handleSearch = async () => {

        if (query.length !== 0) {
            setLoading(true);
            setResponse(null);
            setError('');
            try {
                const res = await axios.post<LLMResponse>('http://localhost:8000/query',
                    {
                        query: query
                    });
                setResponse(res.data);
            } catch (error) {
                console.error("Error fetching data", error);
            } finally {
                setLoading(false);
            }
        }
        else {
            setError('Please enter your query.')
        }

    };

    return (
        <div className="flex flex-col p-4">
            <div className='flex flex-row justify-center'>
                <input
                    type='text'
                    required={true}
                    placeholder='Enter your query'
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className='w-1/2 m-2 p-2 rounded-lg placeholder-gray-400 bg-gray-50 border border-gray-300 focus:outline-none focus:ring focus:ring-blue-300'
                />
                <p className='text-red-500 m-2'>{error}</p>
                <button
                    onClick={handleSearch}
                    className="px-5 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-700 hover:text-white focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
                >
                    Search
                </button>

            </div>


            <div className='flex flex-col items-center m-5'>
                {loading && (
                    <div className="">
                        <Oval color="#00BFFF" height={80} width={80} />
                    </div>
                )}
                {response && (
                    <div className="text-left">
                        <h3 className="font-bold">Response</h3>
                        <p>{response.response}</p>
                        <h3 className="font-bold">Sources</h3>
                        <div className="space-y-4">
                            {response.retrieved_nodes.map((node, index) => (
                                <div key={index} className="p-4 border rounded shadow-sm">
                                    <ul className="list-disc pl-5">
                                        {Object.entries(node.metadata).map(([key, value]) => {
                                            if (key === 'Link') {
                                                return (
                                                    <li key={key}><strong>{key}: </strong><a href={value} className='text-blue-300'>Here</a></li>
                                                )
                                            }
                                            return (<li key={key}><strong>{key}:</strong> {value}</li>)
                                        }

                                        )}
                                    </ul>
                                    <h4 className="font-bold">Score:</h4>
                                    <p>{node.score}</p>
                                </div>
                            ))}
                        </div>
                    </div>

                )}
            </div>

        </div>
    );
};

export default SearchBar;
