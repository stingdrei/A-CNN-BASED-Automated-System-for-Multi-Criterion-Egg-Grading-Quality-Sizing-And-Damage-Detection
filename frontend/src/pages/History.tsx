import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FileImage, Film, Trash2, ExternalLink } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/common/Card';
import { apiClient } from '../api/client';
import { Prediction } from '../types';

export function History() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<number | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const data = await apiClient.getPredictions(0, 50);
        setPredictions(data);
      } catch (error) {
        console.error('Failed to fetch predictions:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchPredictions();
  }, []);

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this prediction?')) return;
    
    setDeleting(id);
    try {
      await apiClient.deletePrediction(id);
      setPredictions((prev) => prev.filter((p) => p.id !== id));
    } catch (error) {
      setError(getErrorMessage(error));
    } finally {
      setDeleting(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Prediction History</h1>

      {predictions.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-gray-500">No predictions yet</p>
            <Link
              to="/upload"
              className="mt-4 inline-block px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
            >
              Upload First Image
            </Link>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>{predictions.length} Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">File</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Type</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Status</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Detections</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Time</th>
                    <th className="text-right py-3 px-4 text-sm font-medium text-gray-500">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((pred) => (
                    <tr key={pred.id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4">
                        <Link
                          to={`/result/${pred.id}`}
                          className="text-sm font-medium text-gray-900 hover:text-primary-600"
                        >
                          {pred.file_name}
                        </Link>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          {pred.file_type === 'image' ? (
                            <FileImage className="w-4 h-4 text-gray-400" />
                          ) : (
                            <Film className="w-4 h-4 text-gray-400" />
                          )}
                          <span className="text-sm text-gray-600 capitalize">{pred.file_type}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-2 py-1 text-xs rounded-full ${
                            pred.status === 'completed'
                              ? 'bg-green-100 text-green-700'
                              : pred.status === 'failed'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-yellow-100 text-yellow-700'
                          }`}
                        >
                          {pred.status}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-sm text-gray-600">
                          {pred.total_detections}
                          {pred.damaged_count > 0 && (
                            <span className="text-red-600"> ({pred.damaged_count} damaged)</span>
                          )}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-sm text-gray-500">
                          {pred.processing_time_ms}ms
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center justify-end gap-2">
                          <Link
                            to={`/result/${pred.id}`}
                            className="p-1 text-gray-400 hover:text-primary-600"
                          >
                            <ExternalLink className="w-4 h-4" />
                          </Link>
                          <button
                            onClick={() => handleDelete(pred.id)}
                            disabled={deleting === pred.id}
                            className="p-1 text-gray-400 hover:text-red-600"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}