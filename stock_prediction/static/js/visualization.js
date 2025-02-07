import React, { useState, useEffect } from 'react';
import { 
    LineChart, Line, XAxis, YAxis, CartesianGrid, 
    Tooltip, Legend, ResponsiveContainer,
    AreaChart, Area
} from 'recharts';
import { Card } from '@/components/ui/card';

const HistoricalDataVisualization = ({ data, company }) => {
    const [chartData, setChartData] = useState([]);
    const [volumeData, setVolumeData] = useState([]);

    useEffect(() => {
        if (data && data.length > 0) {
            // Process data for price chart
            const processedData = data.map(item => ({
                date: new Date(item.Date).toLocaleDateString(),
                price: item['Current Price (£)'],
                high: item['High Price (£)'],
                low: item['Low Price (£)'],
                volume: item['Trading Volume']
            }));

            setChartData(processedData);
            setVolumeData(processedData);
        }
    }, [data]);

    return (
        <div className="space-y-6">
            <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">
                    Price History for {company}
                </h3>
                <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                                dataKey="date"
                                tick={{ fontSize: 12 }}
                                interval={'preserveStartEnd'}
                            />
                            <YAxis 
                                yAxisId="left"
                                tick={{ fontSize: 12 }}
                                label={{ 
                                    value: 'Price (£)', 
                                    angle: -90, 
                                    position: 'insideLeft' 
                                }}
                            />
                            <Tooltip />
                            <Legend />
                            <Line 
                                type="monotone" 
                                dataKey="price" 
                                stroke="#8884d8" 
                                name="Current Price"
                                dot={false}
                                yAxisId="left"
                            />
                            <Line 
                                type="monotone" 
                                dataKey="high" 
                                stroke="#82ca9d" 
                                name="High"
                                dot={false}
                                yAxisId="left"
                            />
                            <Line 
                                type="monotone" 
                                dataKey="low" 
                                stroke="#ff7300" 
                                name="Low"
                                dot={false}
                                yAxisId="left"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </Card>

            <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">
                    Trading Volume
                </h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={volumeData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                                dataKey="date"
                                tick={{ fontSize: 12 }}
                                interval={'preserveStartEnd'}
                            />
                            <YAxis 
                                tick={{ fontSize: 12 }}
                                label={{ 
                                    value: 'Volume', 
                                    angle: -90, 
                                    position: 'insideLeft' 
                                }}
                            />
                            <Tooltip />
                            <Area
                                type="monotone"
                                dataKey="volume"
                                stroke="#8884d8"
                                fill="#8884d8"
                                name="Trading Volume"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </Card>
        </div>
    );
};

export default HistoricalDataVisualization;