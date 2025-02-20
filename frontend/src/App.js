import React, { useState } from "react";
import axios from "axios";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function TextClassificationUI() {
  const [text, setText] = useState("");
  const [processedText, setProcessedText] = useState(null);

  const preprocessText = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/preprocess", {
        text: text,
      });
      setProcessedText(response.data.processed_text);
    } catch (error) {
      console.error("Error preprocessing text:", error);
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <Card>
        <CardContent>
          <h2 className="text-xl font-bold mb-4">Text Classification</h2>
          <Input
            placeholder="Enter text for classification"
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="mb-4"
          />
          <Button onClick={preprocessText} className="w-full">
            Preprocess Text
          </Button>
          {processedText && (
            <div className="mt-4 p-4 bg-gray-100 rounded-lg">
              <h3 className="font-semibold">Processed Text: {processedText}</h3>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
