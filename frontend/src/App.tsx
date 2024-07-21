import React from 'react';
import './App.css';
import SearchBar from './components/SearchBar';

const App: React.FC = () => {
    return (
        <div className="App">
            <header className="bg-gray-800 text-white p-4 shadow-md">
                <h1 className="text-2xl font-bold">LLM Based RAG Frontend</h1>
            </header>
            <main className="p-4">
                <SearchBar />
            </main>
        </div>
    );
}

export default App;
