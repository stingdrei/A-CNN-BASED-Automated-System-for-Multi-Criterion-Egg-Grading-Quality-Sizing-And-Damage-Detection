import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FileImage, Film, Target, Clock, AlertTriangle, TrendingUp, Lightbulb, X } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '../components/common/Card';
import { apiClient } from '../api/client';
import { DashboardStats, GradeDistribution } from '../types';

const statCards = [
  { key: 'total_predictions', label: 'Total Predictions', icon: Target, color: 'text-primary-600', bg: 'bg-primary-50', tooltip: 'Total image/video analyses performed' },
  { key: 'total_images', label: 'Images', icon: FileImage, color: 'text-blue-600', bg: 'bg-blue-50', tooltip: 'Number of image files analyzed' },
  { key: 'total_videos', label: 'Videos', icon: Film, color: 'text-purple-600', bg: 'bg-purple-50', tooltip: 'Number of video files analyzed' },
  { key: 'total_detections', label: 'Total Detections', icon: AlertTriangle, color: 'text-amber-600', bg: 'bg-amber-50', tooltip: 'Total eggs detected across all analyses' },
];

export function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [gradeData, setGradeData] = useState<GradeDistribution | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [statsData, gradeDistribution] = await Promise.all([
          apiClient.getDashboardStats(),
          apiClient.getGradeDistribution()
        ]);
        setStats(statsData);
        setGradeData(gradeDistribution);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <div className="p-4 text-sm text-red-600 bg-red-50 rounded-lg">
          {error}
        </div>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <Link
          to="/upload"
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          New Prediction
        </Link>
      </div>

      {/* Quick Start Hint */}
      {showHelp && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Lightbulb className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-blue-900">Quick Start</p>
              <p className="text-sm text-blue-800 mt-1">
                Click <strong>New Prediction</strong> to upload an image → Adjust confidence threshold → View results in History
              </p>
              <button
                onClick={() => setShowHelp(false)}
                className="text-xs text-blue-600 hover:text-blue-800 mt-2 flex items-center gap-1"
              >
                <X className="w-3 h-3" /> Hide this hint
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Grade Distribution */}
      {gradeData && gradeData.total > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-5 h-5" />
              Grade Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'Grade AA', value: gradeData.grade_aa, fill: '#22c55e' },
                    { name: 'Grade A', value: gradeData.grade_a, fill: '#3b82f6' },
                    { name: 'Grade B', value: gradeData.grade_b, fill: '#f59e0b' },
                    { name: 'N/A', value: gradeData.grade_na, fill: '#6b7280' },
                    { name: 'Reject', value: gradeData.grade_reject, fill: '#ef4444' },
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {[
                    { name: 'Grade AA', value: gradeData.grade_aa, fill: '#22c55e' },
                    { name: 'Grade A', value: gradeData.grade_a, fill: '#3b82f6' },
                    { name: 'Grade B', value: gradeData.grade_b, fill: '#f59e0b' },
                    { name: 'N/A', value: gradeData.grade_na, fill: '#6b7280' },
                    { name: 'Reject', value: gradeData.grade_reject, fill: '#ef4444' },
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* How it works & Troubleshooting */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              How It Works
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="text-sm text-gray-600 space-y-2 list-decimal list-inside">
              <li>Upload an image or video on the <Link to="/upload" className="text-primary-600 hover:underline">Upload</Link> page</li>
              <li>YOLO model detects eggs in the image</li>
              <li>View detailed results including detection count and confidence scores</li>
              <li>Download annotated image with bounding boxes</li>
            </ol>
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">
                <strong>Tip:</strong> Use higher confidence (0.8+) for fewer false detections, or lower (0.5) to catch more eggs.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Recent Predictions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {stats.recent_predictions.length === 0 ? (
              <div className="text-center py-4">
                <p className="text-sm text-gray-500">No predictions yet</p>
                <Link to="/upload" className="text-primary-600 hover:underline text-sm mt-2 inline-block">
                  Upload your first image →
                </Link>
              </div>
            ) : (
              <div className="space-y-3">
                {stats.recent_predictions.slice(0, 5).map((pred) => (
                  <Link
                    key={pred.id}
                    to={`/result/${pred.id}`}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <FileImage className="w-5 h-5 text-gray-400" />
                      <span className="text-sm font-medium text-gray-900 truncate max-w-[150px]">
                        {pred.file_name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        pred.status === 'completed' ? 'bg-green-100 text-green-700' :
                        pred.status === 'failed' ? 'bg-red-100 text-red-700' :
                        'bg-yellow-100 text-yellow-700'
                      }`}>
                        {pred.status}
                      </span>
                      <span className="text-xs text-gray-500">
                        {pred.total_detections} detections
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Troubleshooting */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-600" />
              Troubleshooting
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-700">Image won't load?</p>
                <p className="text-gray-500 mt-1">Try a different image format (JPG/PNG)</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-700">No detections?</p>
                <p className="text-gray-500 mt-1">Lower the confidence threshold</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-700">Too many false detections?</p>
                <p className="text-gray-500 mt-1">Increase confidence to 0.8 or higher</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="font-medium text-gray-700">Want to see bounding boxes?</p>
                <p className="text-gray-500 mt-1">Check "Save annotated" when uploading</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}